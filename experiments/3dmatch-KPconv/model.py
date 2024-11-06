import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import OverallLoss, Evaluator, OverlapLoss
from geotransformer import GeoTransformer_local, GeoTransformer_global
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances 
from visualize import *

class Registration(nn.Module):
    def __init__(self, cfg):
        super(Registration, self).__init__()
        self.cfg = cfg
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.node_num = cfg.model.node_num
        self.T = cfg.model.stage_num

        self.patch_size = cfg.patch.patch_size
        self.point_size = cfg.patch.points_size

        self.global_transformer = GeoTransformer_global(cfg)

        self.local_transformer = GeoTransformer_local(cfg)

        self.conf_logits_decoder = nn.Linear(cfg.pos_embed.d_embed, 1) 

        self.overlap_threshold = cfg.model.classifier_threshold  

        self.epsilon = cfg.model.epsilon
        self.min_samples = cfg.model.min_samples

        self.in_proj_m = nn.Linear(cfg.geotransformer.output_dim * 2, cfg.geotransformer.output_dim)
        
        # loss function, evaluator
        self.loss_func = OverallLoss(cfg).cuda()
        self.loss_classifier = OverlapLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

        self.weight_overlap_loss = cfg.loss.weight_overlap_loss
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss
        self.weight_transform_loss = cfg.loss.weight_transform_loss

        self.view_point = torch.tensor([0., 0., 0.]).cuda()

    def forward(self, data_dict):
        loss_dict, eva_dict, transform_dict = {}, {}, {}
        total_loss_list = []

        # Downsample point clouds
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][2][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][2].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        # 1.1 Overall sample and Global Encoder
        output_dict, feats_list, ref_feats_c, src_feats_c, estimated_transform, \
        ref_node_corr_indices, src_node_corr_indices, \
        ref_point_corr_indices, src_point_corr_indices = self.global_transformer(ref_points_c, src_points_c, ref_points_f, src_points_f, transform, data_dict=data_dict)  


        select_corr_scores = output_dict['select_corr_scores']
        corr_indices = torch.nonzero(select_corr_scores > 0).squeeze(-1)
        if len(src_point_corr_indices) < 2:
            ref_point_corr_indices = torch.randint(0, ref_length_f-1, (2,)).cuda()
            src_point_corr_indices = torch.randint(0, len(points_f)-ref_length_f-1, (2,)).cuda()
        output_dict['ref_indices'] = ref_point_corr_indices
        output_dict['src_indices'] = src_point_corr_indices
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        feats_f = feats_list[0]
        ref_feats_f, src_feats_f = feats_f[:ref_length_f], feats_f[ref_length_f:]
        ref_corr_feats = ref_feats_f[ref_point_corr_indices]  
        src_corr_feats = src_feats_f[src_point_corr_indices]  

        # 1.2 Overlap prediction
        if self.training:  
            ref_corr_overlap = self.conf_logits_decoder(ref_corr_feats.squeeze(0))
            src_corr_overlap = self.conf_logits_decoder(src_corr_feats.squeeze(0))
            # loss and eval
            output_dict['ref_overlap'] = ref_corr_overlap
            output_dict['src_overlap'] = src_corr_overlap
        
            loss_global_dict = self.loss_func(output_dict, data_dict)
            loss_dict.update(loss_global_dict)
            overlap_loss = self.loss_classifier(output_dict, data_dict)
            loss_dict.update({'overlap_score_0': overlap_loss})
            loss = self.weight_coarse_loss * loss_global_dict[f'coarse_loss_0'] + self.weight_fine_loss * loss_global_dict[f'fine_loss_0'] + self.weight_overlap_loss * overlap_loss#
            loss_dict.update({'loss_0': loss})
            total_loss_list.append(loss)

            self.overlap_threshold = 0.0
            ref_overlap_indices = torch.nonzero(ref_corr_overlap.squeeze(-1) > self.overlap_threshold)
            src_overlap_indices = torch.nonzero(src_corr_overlap.squeeze(-1) > self.overlap_threshold)
            _, ref_point_corr_overlap_indices = common_elements_indices(ref_overlap_indices.reshape(-1), ref_point_corr_indices) 
            _, src_point_corr_overlap_indices = common_elements_indices(src_overlap_indices.reshape(-1), src_point_corr_indices)
            
            point_corr_indices = ref_point_corr_overlap_indices[common_elements_indices(ref_point_corr_overlap_indices, src_point_corr_overlap_indices)[0]]
            if len(point_corr_indices) > 0:
                ref_point_corr_indices = ref_point_corr_indices[point_corr_indices].unique()     # [ref_point_corr_overlap_indices]
                src_point_corr_indices = src_point_corr_indices[point_corr_indices].unique()     # [src_point_corr_overlap_indices]
            else :
                ref_point_corr_indices = ref_point_corr_indices[ref_point_corr_overlap_indices].unique()
                src_point_corr_indices = src_point_corr_indices[src_point_corr_overlap_indices].unique()
        
        result_dict = self.evaluator(output_dict, data_dict, 99)
        eva_dict.update(result_dict)
        loss_dict.update(result_dict)
        # early-exit
        if len(ref_point_corr_indices) == 0 or len(src_point_corr_indices) == 0 :
            print(f'abab_00--error')
            if not self.training:
                return eva_dict, output_dict
            else:
                loss_dict['total_loss'] = sum(total_loss_list)
                return loss_dict   
             
        # # 1.4 classifier 
        p = False
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        corr_scores = output_dict['corr_scores']
        ref_corr_points_transform, src_corr_points_transform, corr_scores_transform = ref_corr_points[corr_indices], src_corr_points[corr_indices], corr_scores[corr_indices] # 
        
        if not self.training:
            renew_fitness_one= classifier(ref_corr_points_transform.unsqueeze(0), src_corr_points_transform.unsqueeze(0))

        # message1 = f'before:renew_fitness_one:{renew_fitness_one.item()}', 
        
        if not self.training and renew_fitness_one < self.overlap_threshold:
            # if result_dict[f'RR_{99}'] < 1.0:
                # print(f'abab_correct')
            # else:
                # print(f'abab_err')
            p = True
            # print(message1)
            # print('****************************************')
        data_save = {
                    "p0": src_points,           "p1": ref_points,
                    "p0_f": src_points_f,       "p1_f": ref_points_f,
                    "gt_transform" :transform
                    }
        
        if not self.training and result_dict[f'RR_{99}'] < 1.0:
            print(f'abab_00')
            # p = True 
        

        if not p and not self.training:
            return eva_dict, output_dict
        
        ref_points, src_points = ref_points_c, src_points_c
        ref_encoder_feats, src_encoder_feats = ref_feats_c, src_feats_c
        src_corr_scores, ref_corr_scores = corr_scores, corr_scores
        ref_corr_scores_transform, src_corr_scores_transform = corr_scores_transform, corr_scores_transform
        contrast_renew_fitness = []

        for patch_step in range(1, self.T):
            local_dict = {}
            local_dict.update(output_dict)
            global_result = result_dict[f'RR_{99}']
            # 2.1 prediction point-to-patch upsample
            if len(ref_point_corr_indices) >1 and len(src_point_corr_indices) > 1 and self.training:
                src_points_state, ref_points_state = src_corr_points[src_point_corr_indices], ref_corr_points[ref_point_corr_indices]
                src_corr, ref_corr = corr_scores[src_point_corr_indices], corr_scores[ref_point_corr_indices]
            elif patch_step != 1 and not self.training:
                src_points_state, ref_points_state = torch.cat((src_corr_points, src_points_state)), torch.cat((ref_corr_points, ref_points_state))
                src_corr, ref_corr = torch.cat((src_corr, src_corr_scores)), torch.cat((ref_corr, ref_corr_scores))
            else:
                src_points_state, ref_points_state = src_corr_points, ref_corr_points
                src_corr, ref_corr = src_corr_scores, ref_corr_scores
            src_points_state, src_corr = select_highest(src_points_state, src_corr)
            ref_points_state, ref_corr = select_highest(ref_points_state, ref_corr)
            ref_points_state, src_points_state = torch.cat((ref_corr_points_transform, ref_points_state)), torch.cat((src_corr_points_transform, src_points_state))
            
            ref_corr = torch.cat((ref_corr, ref_corr_scores_transform)) # if patch_step == 1 else None
            src_corr = torch.cat((src_corr, src_corr_scores_transform)) # if patch_step == 1 else None
            
            nums = min(src_points_state.shape[0], ref_points_state.shape[0]) if patch_step == 1 else max(src_num_clusters, ref_num_clusters) * 32
            
            ref_points_state_list, ref_overlap_state_list, ref_num_clusters = dbs_cluster(ref_points_state, ref_corr, epsilon=self.epsilon, min_samples=self.min_samples, nums=nums)
            src_points_state_list, src_overlap_state_list, src_num_clusters = dbs_cluster(src_points_state, src_corr, epsilon=self.epsilon, min_samples=self.min_samples, nums=nums)
            if ref_num_clusters > 0 and (len(src_points_state_list) > 0 and len(ref_points_state_list) > 0):
                ref_action_list = [torch.mean(ref_points_state_list[i], 0).unsqueeze(0)  for i in range(ref_num_clusters)]    # *ref_overlap_state_list[i]
                src_action_list = [torch.mean(src_points_state_list[i], 0).unsqueeze(0)  for i in range(src_num_clusters)]    # *src_overlap_state_list[i]
                
                ref_action = torch.cat(ref_action_list).cuda()
                src_action = torch.cat(src_action_list).cuda()
            else:
                ref_action = torch.mean(ref_points_state, 0).unsqueeze(0)
                src_action = torch.mean(src_points_state, 0).unsqueeze(0)
            
            ref_nodes_state, src_nodes_state = ref_points[ref_node_corr_indices], src_points[src_node_corr_indices]
            data_stage_save = {"p0_corr": src_nodes_state,  "p1_corr": ref_nodes_state,
                        "p0_points_corr": src_corr_points, "p1_points_corr": ref_corr_points,
                        "p0_points_corr_transform": src_corr_points_transform,  "p1_points_corr_transform": ref_corr_points_transform,   # 
                        "p0_center": src_action,       "p1_center": ref_action,       
                        "sample_transform" :estimated_transform
                        }
            data_save.update(data_stage_save) 

            # if ref_num_clusters <5 or src_num_clusters<5 :
            #     ref_action = ref_points if ref_num_clusters <5 else ref_action
            #     src_action = src_points if src_num_clusters <5 else src_action
            #     print(f'abab_{patch_step}--add')
                

            ref_node_indices_action = get_patch(len(ref_points), ref_points_f, ref_action, nearest=True if not self.training else False) 
            src_node_indices_action = get_patch(len(src_points), src_points_f, src_action, nearest=True if not self.training else False) 

            ref_points, src_points = ref_points_f[ref_node_indices_action], src_points_f[src_node_indices_action] 
            ref_feats, src_feats = ref_feats_f[ref_node_indices_action], src_feats_f[src_node_indices_action] 
            
            if len(ref_points) <5 or len(src_points)<5 :
                print(f'abab_{patch_step}--error')
                break

            local_state_dict, _, ref_encoder_feats, src_encoder_feats, estimated_transform, \
            ref_node_corr_indices, src_node_corr_indices,  \
            ref_point_corr_indices, src_point_corr_indices = self.local_transformer(ref_points, src_points, ref_points_f, src_points_f, transform, \
                                                                                    ref_feats, src_feats, ref_feats_f, src_feats_f)   #
            local_dict.update(local_state_dict)
            
            data_stage_save = {
                         "p0_sample": src_points,       "p1_sample": ref_points,
                         "p0_sample_corr": src_points[src_node_corr_indices],
                         "p1_sample_corr": ref_points[ref_node_corr_indices],
                         "p0_sample_points_corr": local_dict['src_corr_points'],
                         "p1_sample_points_corr": local_dict['ref_corr_points'],
                         }
            data_save.update(data_stage_save) 
            # visualize(data_save) 

            select_corr_scores = local_dict['select_corr_scores']
            corr_indices = torch.nonzero(select_corr_scores > 0).squeeze(-1)
            
            local_dict['ref_indices'] = ref_point_corr_indices  
            local_dict['src_indices'] = src_point_corr_indices
            ref_corr_feats = ref_feats_f[ref_point_corr_indices]      
            src_corr_feats = src_feats_f[src_point_corr_indices]     

            # 2.3 Overlap prediction
            if len(ref_point_corr_indices) <2 or len(src_point_corr_indices)<2 :
                print(f'abab_{patch_step}--error')
                break

            if self.training:  
                ref_corr_overlap = self.conf_logits_decoder(ref_corr_feats.squeeze(0))
                src_corr_overlap = self.conf_logits_decoder(src_corr_feats.squeeze(0))             
                # loss and eval
                local_dict['ref_overlap'] = ref_corr_overlap
                local_dict['src_overlap'] = src_corr_overlap

                loss_local_dict = self.loss_func(local_dict, data_dict, patch_step)
                loss_dict.update(loss_local_dict)
                
                overlap_loss_local = self.loss_classifier(local_dict, data_dict)
                loss_dict.update({f'overlap_score_{patch_step}': overlap_loss_local})
                loss = self.weight_coarse_loss * loss_local_dict[f'coarse_loss_{patch_step}'] + self.weight_fine_loss * loss_local_dict[f'fine_loss_{patch_step}']  + self.weight_overlap_loss * overlap_loss_local 
                loss_dict.update({f'loss_{patch_step}': loss})
                total_loss_list.append(loss)

                self.overlap_threshold = 0.0
                ref_overlap_indices = torch.nonzero(ref_corr_overlap.squeeze(-1) > self.overlap_threshold)
                src_overlap_indices = torch.nonzero(src_corr_overlap.squeeze(-1) > self.overlap_threshold)
                _, ref_point_corr_overlap_indices = common_elements_indices(ref_overlap_indices.reshape(-1), ref_point_corr_indices)  
                _, src_point_corr_overlap_indices = common_elements_indices(src_overlap_indices.reshape(-1), src_point_corr_indices)
                
                point_corr_indices = ref_point_corr_overlap_indices[common_elements_indices(ref_point_corr_overlap_indices, src_point_corr_overlap_indices)[0]]
                if len(point_corr_indices) > 0:
                    ref_point_corr_indices = ref_point_corr_indices[point_corr_indices].unique()     # [ref_point_corr_overlap_indices]
                    src_point_corr_indices = src_point_corr_indices[point_corr_indices].unique()     # [src_point_corr_overlap_indices]
                else :
                    ref_point_corr_indices = ref_point_corr_indices[ref_point_corr_overlap_indices].unique()
                    src_point_corr_indices = src_point_corr_indices[src_point_corr_overlap_indices].unique()
            
            select_corr_scores = local_dict['select_corr_scores']
            corr_indices = torch.nonzero(select_corr_scores > 0).squeeze(-1)
            
            # 2.6 classifier
            ref_corr_points = local_dict['ref_corr_points']       
            src_corr_points = local_dict['src_corr_points']
            corr_scores = local_dict['corr_scores']
            ref_corr_points_transform, src_corr_points_transform = ref_corr_points[corr_indices], src_corr_points[corr_indices]
            ref_corr_scores_transform, src_corr_scores_transform = corr_scores[corr_indices], corr_scores[corr_indices]
            src_corr_scores, ref_corr_scores = corr_scores, corr_scores 

            data_stage_iter_save = {"p0_points_corr_transform_iter": src_corr_points_transform,  "p1_points_corr_transform_iter": ref_corr_points_transform,}
            data_save.update(data_stage_iter_save)
            
            # ####################
            if not self.training:
                renew_fitness_one_before = renew_fitness_one
                renew_fitness_one = classifier(ref_corr_points_transform.unsqueeze(0), src_corr_points_transform.unsqueeze(0))
                # message1 = f'before:renew_fitness_one:{renew_fitness_one_before.item()}', 
                # message2 = f' new :renew_fitness_one:{renew_fitness_one.item()}', 
                contrast_renew_fitness.append(renew_fitness_one_before-renew_fitness_one)
                # contrast_message = f'contrast:renew_fitness_one:{(renew_fitness_one-renew_fitness_one_before).item()}', 
                if patch_step == 1 and contrast_renew_fitness[0] >= 30:
                    break
                if patch_step == 2 and contrast_renew_fitness[1] > 0:
                    break
                if patch_step == 3 and contrast_renew_fitness[2] > -30:
                    break

            result_dict = self.evaluator(local_dict, data_dict, 99)
            eva_dict.update(result_dict)
            loss_dict.update(result_dict)     
            p = False

            if not self.training and result_dict[f'RR_{99}'] < 1.0:
                print(f'abab_{patch_step}')
                if global_result == 1.0:
                    print(f'abab_bad_{patch_step}')
                # print(message1)
                # print(message2)
                # print(contrast_message)
                # print('-----------------------------------')
            # else:
            #     if global_result < 1.0:
            #         print(f'abab_good_{patch_step}')   
            #     else:
            #         print(f'abab_still')
            #     # print(message1)
            #     # print(message2)
            #     # print(contrast_message)
            #     # print('-----------------------------------') 
            
            output_dict = local_dict
            # if not p and not self.training:
            #     return eva_dict, local_dict
        
        if self.training:
            loss = sum(total_loss_list) / self.T
            loss_dict['total_loss'] = loss
            
            return loss_dict
        
        else:
            return eva_dict, output_dict


def get_patch(point_size, origin, action_sequence, nearest=False, max_patch_size=None, patch_radius=0.25):
    """Get small patch of the original point cloud"""
    if max_patch_size is None:
        max_patch_size = min(256, point_size)
 
    patch_coordinate = action_sequence

    dist_mat = pairwise_distance(patch_coordinate, origin) # (M, N)
    if not nearest:
        dist_sequence = dist_mat.view(-1)
        
        p = False
        while p == False:
            knn_indices = torch.arange(0, len(dist_sequence)).cuda()[dist_sequence < patch_radius]
            if len(knn_indices) > max_patch_size:
                knn_indices = dist_sequence.topk(k=max_patch_size, dim=0, largest=False)[1]  
                p = True
            else:
                patch_radius *= 1.1
            
        knn_colum = knn_indices % dist_mat.shape[1]
        knn_row = knn_indices % dist_mat.shape[0]
        unique_elements = knn_colum  # patch_coordinate, knn_colum
    else:
        unique_elements, counts = torch.unique(torch.min(dist_mat, dim=1)[1], return_counts=True)
    return unique_elements

def common_elements_indices(tensor1, tensor2):
    # Use unsqueeze to add a dimension to facilitate the use of the torch.eq function 
    tensor1_unsqueezed = tensor1.unsqueeze(0)
    tensor2_unsqueezed = tensor2.unsqueeze(1)

    # Find the mask of common elements in two tensors
    common_elements = torch.eq(tensor1_unsqueezed, tensor2_unsqueezed)

    # Use torch.nonzero to find the position index of these common elements in the first tensor
    indices_tensor1 = torch.nonzero(common_elements)[:, 1]
    indices_tensor2 = torch.nonzero(common_elements)[:, 0]

    return indices_tensor1, indices_tensor2

def dbs_cluster(points, weights=None, epsilon=0.25, min_samples=3, nums=None):
    points = points.cpu().detach().numpy()
    if weights is not None:
        weights = weights.cpu().detach().numpy()
    
    cluster_count = 0  
    target_min, target_max = 0.5, 1
    nums = len(points) if nums is None else nums
    # DBSCAN
    while True:
        if weights is not None:
            min_val, max_val = min(weights), max(weights)
            normalized_weights = [(target_max - target_min) * (w - min_val) / (max_val-min_val) + target_min for w in weights]
            
            distances = pairwise_distances(points) 
            distances /= np.array(normalized_weights, dtype=np.float32).reshape(-1, 1) 
            db = DBSCAN(eps=epsilon, min_samples=min_samples, metric='precomputed') 
            labels = db.fit_predict(distances) 
        else:
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
            labels = dbscan.fit_predict(points)

        num_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        
        if num_clusters > nums / 32:
            
            epsilon *= 1.1
            min_samples += 1
            cluster_count += 1
        elif num_clusters < 5 and cluster_count < 5:
            epsilon /= 1.2
            min_samples = max(2, min_samples - 1)
            cluster_count += 1
        else:
            break
    
    points_cluster = []
    feats_cluster = []
    overlap_cluster = []
    for i in range(num_clusters): 
        indices = np.where(labels==i)[0]
        points_cluster.append(torch.tensor(points[indices]))
        if weights is not None:
            overlap_cluster.append(torch.tensor(weights[indices]))

    return points_cluster, overlap_cluster, num_clusters


########################################################################################

def classifier(tgt_keypts, src_keypts, trans=None, weights=None):
    inlier_threshold = 0.10   # 0.10 for 3DMatch
    FS_TCD_thre = 0.05 

    src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
    target_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
    cross_dist = src_dist - target_dist
    abs_compatibility = torch.abs(cross_dist)#  

    SC_thre = FS_TCD_thre
    num_corr = src_dist.shape[1]
    corr_compatibility_2 = (abs_compatibility < SC_thre).float()
    compatibility_num_one = torch.sum(corr_compatibility_2, -1)
    renew_fitness_one = compatibility_num_one.max()    
    
    return renew_fitness_one  
#################################################################

def select_highest(corr_points, corr_scores):
    data =  torch.cat((corr_points, corr_scores.unsqueeze(1)), dim=1)
    indices_re = torch.argsort(corr_scores, descending=True)
    data_uni = data[indices_re]
    data_uni_points = torch.unique(corr_points[indices_re], dim=0)
    indices = [torch.nonzero(torch.all(data_uni[:, :-1]==point_uni, dim=1))[0].item() for point_uni in data_uni_points]
    return data_uni[indices, :-1], data_uni[indices, -1]

def remove_repeat(ref_corr_points_transform, src_corr_points_transform, corr_scores_transform):
    ref_data =  torch.cat((ref_corr_points_transform, corr_scores_transform.unsqueeze(1)), dim=1)
    ref_indices_re = torch.argsort(ref_data[:, 0], descending=True)
    ref_data_uni = torch.unique(ref_corr_points_transform[ref_indices_re], dim=0)
    ref_indices_uni = [torch.nonzero(torch.all(ref_corr_points_transform==ref_point_uni, dim=1))[0].item() for ref_point_uni in ref_data_uni]
    src_corr_points_uni, ref_corr_points_uni, corr_scores_uni = src_corr_points_transform[ref_indices_uni], ref_corr_points_transform[ref_indices_uni], corr_scores_transform[ref_indices_uni]

    src_data = torch.cat((src_corr_points_uni, corr_scores_uni.unsqueeze(1)), dim=1)
    src_indices_re = torch.argsort(src_data[:, 0], descending=True)
    src_data_uni = torch.unique(src_corr_points_uni[src_indices_re], dim=0)
    src_indices_uni = [torch.nonzero(torch.all(src_corr_points_uni==src_point_uni, dim=1))[0].item() for src_point_uni in src_data_uni]

    src_corr_points_uni = src_corr_points_uni[src_indices_uni]
    ref_corr_points_uni = ref_corr_points_uni[src_indices_uni]
    corr_scores_uni = corr_scores_uni[src_indices_uni]
    return src_corr_points_uni, ref_corr_points_uni, corr_scores_uni# corr_indices_uni

#########################################################################
def main():
    from config import make_cfg

    cfg = make_cfg()
    model = Registration(cfg) 
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()