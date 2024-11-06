import torch
import torch.nn as nn

from modules.loss.circle_loss import WeightedCircleLoss
from modules.ops.pairwise_distance import pairwise_distance
from modules.registration.metrics import isotropic_transform_error
from modules.ops.transformation import apply_transform


import os
import os.path as osp
from modules.datasets.registration.threedmatch.utils import compute_transform_error, get_gt_logs_and_infos, get_num_fragments


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss

class OverlapLoss(nn.Module):
    def __init__(self, cfg):
        super(OverlapLoss, self).__init__()
        
        self.overlap_criterion = nn.BCEWithLogitsLoss()
    

    def forward(self, output_dict, data_dict):

        src_overlap = data_dict['src_overlap']
        ref_overlap = data_dict['ref_overlap']
        ref_length_f = data_dict['lengths'][2][0].item()

        
        n_pyr = len(data_dict['points'])

        overlap_pyr = {'pyr_0': torch.cat((ref_overlap, src_overlap), dim=0).type(torch.float)}
        invalid_indices = [sum(s) for s in data_dict['lengths']]
        for p in range(1, n_pyr):
            pooling_indices = data_dict['subsampling'][p - 1].clone()
            valid_mask = pooling_indices < invalid_indices[p - 1].item()
            pooling_indices[~valid_mask] = 0

            # Average pool over indices
            overlap_gathered = overlap_pyr[f'pyr_{p-1}'][pooling_indices] * valid_mask
            overlap_gathered = torch.sum(overlap_gathered, dim=1) / torch.sum(valid_mask, dim=1)
            overlap_gathered = torch.clamp(overlap_gathered, min=0, max=1)
            overlap_pyr[f'pyr_{p}'] = overlap_gathered

        ref_overlap_f_gt = overlap_pyr[f'pyr_{2}'][:ref_length_f]
        src_overlap_f_gt = overlap_pyr[f'pyr_{2}'][ref_length_f:]
        all_overlap_gt = torch.cat((ref_overlap_f_gt[output_dict['ref_indices']], src_overlap_f_gt[output_dict['src_indices']]))

        all_overlap_pred = torch.cat((output_dict['ref_overlap'], output_dict['src_overlap']), dim=-2)

        loss = self.overlap_criterion(all_overlap_pred[:, 0], all_overlap_gt)
        
        return loss

class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_overlap_loss = cfg.loss.weight_overlap_loss
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss

    def forward(self, output_dict, data_dict, stage=0):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        return {
            f'coarse_loss_{stage}': coarse_loss,
            f'fine_loss_{stage}': fine_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold
        self.gt_indices, self.gt_infos = dict(), dict()

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)   
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()   
        return precision   

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        points = data_dict['points'].detach()
        ref_length = data_dict['lengths'][0][0].item()
        src_points = points[ref_length:]

        rre, rte = isotropic_transform_error(transform, est_transform)

        if not self.gt_indices:
            realignment_transform = torch.matmul(torch.inverse(transform), est_transform)
            realigned_src_points_f = apply_transform(src_points, realignment_transform)
            rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
            recall = torch.lt(rmse, self.acceptance_rmse).float()
        else:
            # eval.py
            scene = data_dict['scene_name']
            ref_frame, src_frame = data_dict['ref_frame'], data_dict['src_frame']
            index = self.gt_indices[scene][ref_frame, src_frame]
            covariance = self.gt_infos[scene][index]['covariance']
            error = compute_transform_error(transform.cpu().numpy(), covariance, est_transform.cpu().numpy())   # 
            accepted = error < self.acceptance_rmse ** 2
            rmse = torch.sqrt(torch.tensor(error)).cuda()
            recall = float(accepted)

        return rre, rte, rmse, recall

    def forward(self, output_dict, data_dict, stage=0):
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict)

        return {
            f'PIR_{stage}': c_precision,
            f'IR_{stage}': f_precision,
            f'RRE_{stage}': rre,
            f'RTE_{stage}': rte,
            f'RMSE_{stage}': rmse,
            f'RR_{stage}': recall,
        }
