import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import OverallLoss, Evaluator, OverlapLoss
from geotransformer import GeoTransformer_local, GeoTransformer_global
from utils.torch import release_cuda
from utils.common import ensure_dir, get_log_string, normal_redirect
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics.pairwise import pairwise_distances 
from visualize import *
import pdb
import time, math

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
        
        self.overlap_threshold = cfg.model.classifier_threshold  

        self.epsilon = cfg.model.epsilon
        if self.epsilon == 0.125:
            self.epsilon = [0.125, 0.25, 0.375, 0.5]
            self.compare_contrast_thre = [20, 0, -20, -40]
        else:
            self.epsilon = [0.3, 0.35, 0.4, 0.45]
            self.compare_contrast_thre = [15, 25, 35, 45]
        self.min_samples = cfg.model.min_samples
        
        # loss function, evaluator
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss

        self.view_point = torch.tensor([0., 0., 0.]).cuda()

    def forward(self, data_dict):
        loss_dict, eva_dict, transform_dict = {}, {}, {}
        total_loss_list = []

        # Downsample point clouds
        transform = data_dict['transform'].detach()
        # import pdb
        # pdb.set_trace()
        ref_length = data_dict['lengths'][0].item()
        points = data_dict['points'].detach()


        ref_points, src_points = points[:ref_length], points[ref_length:]
        ref_feats, src_feats = data_dict['features'][:ref_length], data_dict['features'][ref_length:]
        ref_normals, src_normals = data_dict['normals'][:ref_length], data_dict['normals'][ref_length:]

        # Normal estimation
        # o3d_src_pcd = to_o3d_pcd(src_points)
        # o3d_ref_pcd = to_o3d_pcd(ref_points)
        # o3d_src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        # src_normals = torch.from_numpy(np.asarray(o3d_src_pcd.normals).astype(np.float32)).cuda()
        # src_normals = normal_redirect(src_points, src_normals, view_point=self.view_point)
        # o3d_ref_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        # ref_normals = torch.from_numpy(np.asarray(o3d_ref_pcd.normals).astype(np.float32)).cuda()
        # ref_normals = normal_redirect(ref_points, ref_normals, view_point=self.view_point)

        # 1.1 Overall sample and Global Encoder
        output_dict, feats_list, ref_feats_c, src_feats_c, estimated_transform, \
        ref_node_corr_indices, src_node_corr_indices, \
        ref_point_corr_indices, src_point_corr_indices, data_dict = self.global_transformer(src_points, ref_points, src_feats, ref_feats, transform, src_normals, ref_normals, data_dict=data_dict)    # , training=self.training

        ref_length_f, src_length_f = data_dict['lengths'][2][0].item(), data_dict['lengths'][2][1].item()
        ref_points_c, src_points_c = data_dict['points_list'][-1]
        ref_points_f, src_points_f = data_dict['points_list'][2]
        ref_points_m, src_points_m = data_dict['points_list'][1]


        select_corr_scores = output_dict['select_corr_scores']
        corr_indices = torch.nonzero(select_corr_scores > 0).squeeze(-1)
        # ref_point_corr_indices, src_point_corr_indices = ref_point_corr_indices[corr_indices], src_point_corr_indices[corr_indices]
        if len(src_point_corr_indices) < 2:
            ref_point_corr_indices = torch.randint(0, ref_length_f-1, (2,)).cuda()
            src_point_corr_indices = torch.randint(0, src_length_f-1, (2,)).cuda()
        output_dict['ref_indices'] = ref_point_corr_indices
        output_dict['src_indices'] = src_point_corr_indices
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points
        ref_normals_f = output_dict['ref_normals_f']
        src_normals_f = output_dict['src_normals_f']


        ref_feats_f, src_feats_f = feats_list[2]
        ref_corr_feats = ref_feats_f[ref_point_corr_indices]   
        src_corr_feats = src_feats_f[src_point_corr_indices]   
        ref_corr_normals = ref_normals_f[ref_point_corr_indices] 
        src_corr_normals = src_normals_f[src_point_corr_indices]

        # 1.2 Overlap prediction
        if self.training:  
            loss_global_dict = self.loss_func(output_dict, data_dict)
            loss_dict.update(loss_global_dict)
            loss = self.weight_coarse_loss * loss_global_dict[f'coarse_loss_0'] + self.weight_fine_loss * loss_global_dict[f'fine_loss_0']# + self.weight_overlap_loss * overlap_loss#
            loss_dict.update({'loss_0': loss})
            total_loss_list.append(loss)
        
        result_dict = self.evaluator(output_dict, data_dict, 99)
        eva_dict.update(result_dict)
        loss_dict.update(result_dict)
        # # 提前结束
        if len(ref_point_corr_indices) <= 2 or len(src_point_corr_indices) <= 2 :
            print(f'abab_00--error')
            if not self.training:
                return eva_dict, output_dict
            else:
                loss_dict['total_loss'] = sum(total_loss_list)
                return loss_dict    
        # # 1.4 classifier ### 设置阈值取部分保留
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        corr_scores = output_dict['corr_scores']
        ref_corr_points_transform, src_corr_points_transform, corr_scores_transform = ref_corr_points[corr_indices], src_corr_points[corr_indices], corr_scores[corr_indices]
        ref_corr_normals_transform, src_corr_normals_transform = ref_corr_normals[corr_indices], src_corr_normals[corr_indices]
        transform_dict['estimated_transform'] = estimated_transform
        
        p = False
        if not self.training:
            renew_fitness_one= classifier(ref_corr_points_transform.unsqueeze(0), src_corr_points_transform.unsqueeze(0), ref_corr_normals_transform, src_corr_normals_transform, estimated_transform)

            message1 = f'before:renew_fitness_one:{renew_fitness_one.item()}', 

            if renew_fitness_one < self.overlap_threshold:
                if result_dict[f'RR_{99}'] < 1.0:
                    print(f'abab_correct')
                else:
                    print(f'abab_bad')
                p = True
                print(message1)
                print('****************************************')
        
        #### debug
        # debug = True #  False

        data_save = {
                    "p0": src_points,           "p1": ref_points,
                    "p0_f": src_points_f,       "p1_f": ref_points_f,
                    "gt_transform" :transform, "sample_transform" :estimated_transform
                    }
        # visualize(data_save)
        
        if not self.training and result_dict[f'RR_{99}'] < 1.0:
            print(f'abab_00')
        #     p = True 
        

        if not p and not self.training:
            return eva_dict, output_dict
        
        ref_points, src_points = ref_points_c, src_points_c
        ref_encoder_feats, src_encoder_feats = ref_feats_c, src_feats_c
        contrast_renew_fitness = []


        for patch_step in range(1, self.T):
            local_dict = {}
            local_dict.update(output_dict)
            global_result = result_dict[f'RR_{99}']
            # 提前结束
            if len(corr_indices) <2 or len(ref_corr_points) < 2 or len(src_corr_points) <2:
                break

            src_corr_scores, ref_corr_scores = corr_scores, corr_scores
            ref_corr_scores_transform, src_corr_scores_transform = corr_scores_transform, corr_scores_transform
            # 2.1 prediction point-to-patch upsample
            if patch_step > 1 :    # and not self.training
                src_points_state, ref_points_state = torch.cat((src_corr_points_transform, src_points_cat)), torch.cat((ref_corr_points_transform, ref_points_cat)) # src_points_state  ref_points_state
                src_scores_state, ref_scores_state = torch.cat((src_scores_cat, src_corr_scores_transform)), torch.cat((ref_scores_cat, ref_corr_scores_transform))
            else:
                src_points_state, ref_points_state = src_corr_points, ref_corr_points
                src_scores_state, ref_scores_state = src_corr_scores, ref_corr_scores
            src_points_state_sel, src_scores_state_sel = select_highest(src_points_state, src_scores_state) # src_points_state, src_scores_state # 
            ref_points_state_sel, ref_scores_state_sel = select_highest(ref_points_state, ref_scores_state) # ref_points_state, ref_scores_state # 

            ref_points_cat, src_points_cat = torch.cat((ref_corr_points_transform, ref_points_state_sel)), torch.cat((src_corr_points_transform, src_points_state_sel))
            src_scores_cat = torch.cat((src_corr_scores_transform, src_scores_state_sel)) # if patch_step == 1 else None
            ref_scores_cat = torch.cat((ref_corr_scores_transform, ref_scores_state_sel)) # if patch_step == 1 else None
            
            nums = min(src_points_cat.shape[0], ref_points_cat.shape[0]) if patch_step == 1 else max(src_num_clusters, ref_num_clusters) * 32

            if patch_step==1:
                epsilon = self.epsilon[0]  
            elif patch_step==2:
                epsilon = self.epsilon[1]  # 3D: 0.25  # 3DLo: 0.35  
            elif patch_step==3:
                epsilon = self.epsilon[2]  # 3D: 0.375 # 3DLo: 0.4
            else:
                epsilon = self.epsilon[3]  # 3D: 0.5   # 3DLo: 0.45
            ref_points_state_list, ref_overlap_state_list, ref_num_clusters = dbs_cluster(ref_points_cat, ref_scores_cat, epsilon=epsilon, min_samples=self.min_samples, nums=nums)  # # ref_points_overlap
            src_points_state_list, src_overlap_state_list, src_num_clusters = dbs_cluster(src_points_cat, src_scores_cat, epsilon=epsilon, min_samples=self.min_samples, nums=nums)  # 

            if ref_num_clusters > 0 and (len(src_points_state_list) > 0 and len(ref_points_state_list) > 0):
                ref_action = ref_points_state[torch.randperm(ref_points_state.shape[0])[:ref_num_clusters]]
                src_action = src_points_state[torch.randperm(src_points_state.shape[0])[:src_num_clusters]]
            else:
                ref_action = torch.mean(ref_points_state, 0).unsqueeze(0)
                src_action = torch.mean(src_points_state, 0).unsqueeze(0)
                

            ref_nodes_state, src_nodes_state = ref_points[ref_node_corr_indices], src_points[src_node_corr_indices]
            data_stage_save = {"p0_corr": src_nodes_state,  "p1_corr": ref_nodes_state,
                        "p0_points_corr": src_corr_points, "p1_points_corr": ref_corr_points,
                        "p0_points_corr_transform": src_corr_points_transform,  "p1_points_corr_transform": ref_corr_points_transform,   
                        "p0_center": src_action,       "p1_center": ref_action,       
                        "recompute_transform" :estimated_transform
                        }
            data_save.update(data_stage_save)
            # visualize(data_save) 

            ref_node_indices_action = get_patch(len(ref_points), ref_points_f, ref_action, training=self.training, patch_step=patch_step) # ref_patch_size , ref_action_weight   # ref_points_m
            src_node_indices_action = get_patch(len(src_points), src_points_f, src_action, training=self.training, patch_step=patch_step) # src_patch_size , src_action_weight   # src_points_m

            ref_points, src_points = ref_points_f[ref_node_indices_action], src_points_f[src_node_indices_action] # ref_points_m[ref_node_indices], src_points_m[src_node_indices] # 
            ref_feats, src_feats = ref_feats_f[ref_node_indices_action], src_feats_f[src_node_indices_action] # ref_feats_m[ref_node_indices], src_feats_m[src_node_indices] # 
            
            if len(ref_points) <3 or len(src_points)<3 :       # 3DLo: 3 ,3D:5
                print(f'abab_{patch_step}--error')
                break

            local_state_dict, _, ref_encoder_feats, src_encoder_feats, estimated_transform, \
            ref_node_corr_indices, src_node_corr_indices,  \
            ref_point_corr_indices, src_point_corr_indices = self.local_transformer(ref_points, src_points, ref_points_f, src_points_f, transform, \
                                                                                    ref_feats, src_feats, ref_feats_f, src_feats_f, self.global_transformer)   # 
            local_dict.update(local_state_dict)
            
            
            data_stage_save = {
                         "p0_sample": src_points,       "p1_sample": ref_points,
                         "p0_sample_corr": src_points[src_node_corr_indices],
                         "p1_sample_corr": ref_points[ref_node_corr_indices],
                         "p0_sample_points_corr": local_dict['src_corr_points'],
                         "p1_sample_points_corr": local_dict['ref_corr_points'],
                         'recompute_transform': estimated_transform
                         }
            data_save.update(data_stage_save) 

            select_corr_scores = local_dict['select_corr_scores']
            corr_indices = torch.nonzero(select_corr_scores > 0).squeeze(-1)
            
            local_dict['ref_indices'] = ref_point_corr_indices     
            local_dict['src_indices'] = src_point_corr_indices
            ref_corr_feats = ref_feats_f[ref_point_corr_indices]      
            src_corr_feats = src_feats_f[src_point_corr_indices]     
            ref_corr_normals = ref_normals_f[ref_point_corr_indices] 
            src_corr_normals = src_normals_f[src_point_corr_indices]

            # 2.3 Overlap prediction
            if self.training:  
                
                loss_local_dict = self.loss_func(local_dict, data_dict, patch_step)
                loss_dict.update(loss_local_dict)
                
                loss = self.weight_coarse_loss * loss_local_dict[f'coarse_loss_{patch_step}'] + self.weight_fine_loss * loss_local_dict[f'fine_loss_{patch_step}'] # + self.weight_overlap_loss * overlap_loss_local # self.weight_transform_loss * transform_loss_local +
                loss_dict.update({f'loss_{patch_step}': loss})
                total_loss_list.append(loss)
                
            # 2.6 classifier
            ref_corr_points = local_dict['ref_corr_points']       
            src_corr_points = local_dict['src_corr_points']
            corr_scores = local_dict['corr_scores']
            ref_corr_points_transform, src_corr_points_transform, corr_scores_transform = ref_corr_points[corr_indices], src_corr_points[corr_indices], corr_scores[corr_indices]
            ref_corr_normals_transform, src_corr_normals_transform = ref_corr_normals[corr_indices], src_corr_normals[corr_indices]
            
            
            transform_dict['estimated_transform'] = estimated_transform
            data_stage_iter_save = {"p0_points_corr_transform_iter": src_corr_points_transform,  "p1_points_corr_transform_iter": ref_corr_points_transform,}
            data_save.update(data_stage_iter_save)

            if not self.training:
                renew_fitness_one_before = renew_fitness_one
                renew_fitness_one = classifier(ref_corr_points_transform.unsqueeze(0), src_corr_points_transform.unsqueeze(0), ref_corr_normals_transform, src_corr_normals_transform)
                message1 = f'before:renew_fitness_one:{renew_fitness_one_before.item()}', 
                message2 = f' new :renew_fitness_one:{renew_fitness_one.item()}', 
                contrast_renew_fitness.append(renew_fitness_one_before-renew_fitness_one)
                contrast_message = f'contrast_{patch_step}:renew_fitness_one:{(renew_fitness_one-renew_fitness_one_before).item()}', 
                if patch_step == 1 and contrast_renew_fitness[0] > self.compare_contrast_thre[0]:      # 3DLo 15, 3d: 20
                    break
                if patch_step == 2 and contrast_renew_fitness[1] > self.compare_contrast_thre[1]:      # 3DLo 25, 3d: 0
                    break
                if patch_step == 3 and contrast_renew_fitness[2] > self.compare_contrast_thre[2]:     # 3d:-20
                    break
                # if patch_step == 4 and contrast_renew_fitness[3] > self.compare_contrast_thre[3]:     # 3d:-40
                #     break

            result_dict = self.evaluator(local_dict, data_dict, 99)
            eva_dict.update(result_dict)
            loss_dict.update(result_dict)     
            p = False

            if not self.training and result_dict[f'RR_{99}'] < 1.0:
                print(f'abab_{patch_step}')
                # p = True 
                if global_result == 1.0:
                    print(f'abab_bad_{patch_step}')
                # print(message1)
                # print(message2)
                # print(contrast_message)
                # visualize(data_save) 
                print('-----------------------------------')
            else:
                if global_result < 1.0:
                    # visualize(data_save) 
                    print(f'abab_good_{patch_step}')    
                # print(message1)
                # print(message2)
                # print(contrast_message)

            # if not p and not self.training:
            #     return eva_dict, local_dict
            
            output_dict = local_dict   
        
        if self.training:
            loss = sum(total_loss_list) / self.T
            loss_dict['total_loss'] = loss
            
            return loss_dict
        
        else:
            return eva_dict, output_dict


def get_patch(point_size, origin, action_sequence, weight=None, max_patch_size=None, patch_radius=0.25, training=True, patch_step=0):
    """Get small patch of the original point cloud"""
    if max_patch_size is None:
        max_patch_size = min(256, point_size)
 
    patch_coordinate = action_sequence

    dist_mat = pairwise_distance(patch_coordinate, origin) # (M, N)
    if weight:
        weight = torch.tensor(weight, dtype=torch.float).cuda()
        scaled_weights = ((weight / torch.sum(weight))*point_size).int()
        # print(scaled_weights.int())
        knn_colum = []
        for i in range(dist_mat.shape[0]):
            k = scaled_weights[i].item() if scaled_weights[i].item() > 0 else 1
            knn_colum.append(torch.topk(dist_mat[i], k=k, largest=False)[1])
        knn_colum = torch.cat(knn_colum)
    elif training:
        dist_sequence = dist_mat.view(-1)
        
        p = False
        while p == False:
            knn_indices = torch.arange(0, len(dist_sequence)).cuda()[dist_sequence < patch_radius]
            if len(knn_indices) > max_patch_size:
                knn_indices = dist_sequence.topk(k=max_patch_size, dim=0, largest=False)[1]  # (M, K)   每个c中的node对应最近的points索引
                p = True
            else:
                patch_radius *= 1.1
            
        knn_colum = knn_indices % dist_mat.shape[1]
        knn_row = knn_indices % dist_mat.shape[0]
    ########################### For test #########################################
    else:
        # unique_elements = torch.unique(torch.topk(dist_mat, k=2, dim=1, largest=False, sorted=True)[1])
        unique_elements = torch.unique(torch.min(dist_mat, dim=1)[1], return_counts=False) # if patch_step==1 else torch.unique(torch.topk(dist_mat, k=2, dim=1, largest=False, sorted=True)[1])
        knn_colum = unique_elements
    return knn_colum

def common_elements_indices(tensor1, tensor2):
    tensor1_unsqueezed = tensor1.unsqueeze(0)
    tensor2_unsqueezed = tensor2.unsqueeze(1)

    common_elements = torch.eq(tensor1_unsqueezed, tensor2_unsqueezed)

    indices_tensor1 = torch.nonzero(common_elements)[:, 1]
    indices_tensor2 = torch.nonzero(common_elements)[:, 0]

    return indices_tensor1, indices_tensor2

def dbs_cluster(points, weights=None, epsilon=0.25, min_samples=3, nums=None):
    points = points.cpu().detach().numpy()
    if weights is not None:
        weights = weights.cpu().detach().numpy()
    cluster_count = 0 
    target_min, target_max = 0.1, 1
    nums = len(points) if nums is None else nums
    if epsilon in [0.125, 0.25, 0.375, 0.5]:
        num_min = 32
        cluster_min=5
    else:
        num_min = 4
        cluster_min=3

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
        
        if num_clusters > nums / num_min:         # 3DLo: 4,     3D:32
            epsilon *= 1.1
            min_samples += 1
            cluster_count += 1
        elif num_clusters < cluster_min and cluster_count < 5:        # 3DLo:3 ,3D:5
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


def classifier(tgt_keypts, src_keypts, tgt_normals, src_normals, trans=None):
    SC_thre = 0.05  

    src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
    target_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
    cross_dist = src_dist - target_dist
    abs_compatibility = torch.abs(cross_dist)#  

    corr_compatibility = (abs_compatibility < SC_thre).float()
    compatibility_num_one = torch.sum(corr_compatibility, -1)[0]  
    renew_fitness_one, renew_indices = compatibility_num_one.max(dim=0)         

    return renew_fitness_one
#################################################################

def cal_leading_eigenvector(M, method='power'):
    """
    Calculate the leading eigenvector using power iteration algorithm or torch.symeig
    Input:
        - M:      [bs, num_corr, num_corr] the compatibility matrix
        - method: select different method for calculating the learding eigenvector.
    Output:
        - solution: [bs, num_corr] leading eigenvector
    """
    num_iterations = 10
    if method == 'power':
        # power iteration algorithm
        leading_eig = torch.ones_like(M[:, :, 0:1])
        leading_eig_last = leading_eig
        for i in range(num_iterations):
            leading_eig = torch.bmm(M, leading_eig)
            leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
            if torch.allclose(leading_eig, leading_eig_last):
                break
            leading_eig_last = leading_eig
        leading_eig = leading_eig.squeeze(-1)
        return leading_eig
    elif method == 'eig':  # cause NaN during back-prop
        e, v = torch.symeig(M, eigenvectors=True)
        leading_eig = v[:, :, -1]
        return leading_eig
    else:
        exit(-1)

def pick_seeds(dists, scores, R, max_num):
    """
    Select seeding points using Non Maximum Suppression. (here we only support bs=1)
    Input:
        - dists:       [bs, num_corr, num_corr] src keypoints distance matrix
        - scores:      [bs, num_corr]     initial confidence of each correspondence
        - R:           float              radius of nms
        - max_num:     int                maximum number of returned seeds
    Output:
        - picked_seeds: [bs, num_seeds]   the index to the seeding correspondences
    """
    assert scores.shape[0] == 1

    # parallel Non Maximum Suppression (more efficient)
    score_relation = scores.T >= scores  # [num_corr, num_corr], save the relation of leading_eig
    # score_relation[dists[0] >= R] = 1  # mask out the non-neighborhood node
    score_relation = score_relation.bool() | (dists[0] >= R).bool()
    is_local_max = score_relation.min(-1)[0].float()

    score_local_max = scores * is_local_max
    sorted_score = torch.argsort(score_local_max, dim=1, descending=True)

    # max_num = scores.shape[1]

    return_idx = sorted_score[:, 0: max_num].detach()

    return return_idx
####################################################################
def select_highest(corr_points, corr_scores):
    data =  torch.cat((corr_points, corr_scores.unsqueeze(1)), dim=1)
    indices_re = torch.argsort(corr_scores, descending=True)
    data_uni = data[indices_re]
    data_uni_points = torch.unique(corr_points[indices_re], dim=0)
    indices = [torch.nonzero(torch.all(data_uni[:, :-1]==point_uni, dim=1))[0].item() for point_uni in data_uni_points]
    return data_uni[indices, :-1], data_uni[indices, -1]


#################################################################
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
    corr_indices_uni = [ref_indices_uni[index] for index in src_indices_uni]
    return src_corr_points_uni, ref_corr_points_uni, corr_indices_uni

#########################################################################
def main():
    from config import make_cfg

    cfg = make_cfg()
    model = Registration(cfg)  # create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()