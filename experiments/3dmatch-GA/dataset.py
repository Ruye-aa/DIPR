import os
import random
from functools import partial
import numpy as np
import pickle
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict
import os.path as osp
from utils import random_sample_rotation, random_sample_rotation_v2, get_transform_from_rotation_translation
from utils.registration import get_correspondences
from utils.torch import reset_seed_worker_init_fn
from visualize import *
from utils.common import uniform_sample_rotation, np_get_transform_from_rotation_translation, get_overlap_mask

def train_valid_data_loader(cfg, distributed):
    train_dataset = ThreeDMatchPairDataset(                
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )
    valid_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        'val',
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )

    train_loader = build_dataloader_stack_mode(             
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )
    valid_loader = build_dataloader_stack_mode(            
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
    )
    return train_loader, valid_loader

def test_data_loader(cfg, benchmark):
    test_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        benchmark,
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )
    test_loader = build_dataloader_stack_mode(            
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        )

    return test_loader# , neighbor_limits


class ThreeDMatchPairDataset(torch.utils.data.Dataset):        
    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        overlap_threshold=None,            
        return_corr_indices=False,
        matching_radius=None,
        rotated=False,
    ):
        super().__init__()               

        self.dataset_root = dataset_root
        self.metadata_root = os.path.join(self.dataset_root, 'metadata')
        self.data_root = os.path.join(self.dataset_root, 'data')

        self.subset = subset
        self.point_limit = point_limit              
        self.overlap_threshold = overlap_threshold  
        with open(os.path.join(self.metadata_root, f'{subset}.pkl'), 'rb') as f:
            self.metadata_list = pickle.load(f)     
            if self.overlap_threshold is not None:  
                self.metadata_list = [x for x in self.metadata_list if x['overlap'] > self.overlap_threshold]
        self.rotated = rotated                      

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        self.search_voxel_size = 0.0375 
        self.view_point = np.asarray([0., 0., 0.])
        

    def __len__(self):
        return len(self.metadata_list)

    def _load_point_cloud(self, file_name):

        pcd_path_parts = file_name.split('/')
        pcd_last_parts = pcd_path_parts[-1].split('_')
        pcd_last_parts[0] = 'normal'
        pcd_path_parts[-1] = '_'.join(pcd_last_parts)
        normal_path = '/'.join(pcd_path_parts)
        # normals = torch.load(os.path.join(self.data_root, normal_path))

        points = torch.load(os.path.join(self.data_root, file_name))
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[:self.point_limit]  
            points = points[indices]
            # normals = normals[indices]
            
        return points# , normals
    

    def _augment_point_cloud(self, ref_points, src_points, rotation, translation):
        r"""Augment point clouds.

                ref_points = src_points @ rotation.T + translation

                1. Random rotation to one point cloud.
                2. Random noise.
                """
        aug_rotation = random_sample_rotation(self.aug_rotation)  
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T) 
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)  
            rotation = np.matmul(rotation, aug_rotation.T)

        return ref_points, src_points, rotation, translation

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata: Dict = self.metadata_list[index]
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['ref_frame'] = metadata['frag_id0']
        data_dict['src_frame'] = metadata['frag_id1']
        data_dict['overlap'] = metadata['overlap']

        # get transformation
        rotation = metadata['rotation']
        translation = metadata['translation']

        # get point cloud
        ref_points = self._load_point_cloud(metadata['pcd0'])   # , ref_normals
        src_points = self._load_point_cloud(metadata['pcd1'])   # , src_normals

        # augmentation
        if self.use_augmentation:                            
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation
            )
        if self.rotated:
            ref_rotation = random_sample_rotation_v2()       
            ref_points = np.matmul(ref_points, ref_rotation.T)
            rotation = np.matmul(ref_rotation, rotation)
            translation = np.matmul(ref_rotation, translation)

            src_rotation = random_sample_rotation_v2()
            src_points = np.matmul(src_points, src_rotation.T)
            rotation = np.matmul(rotation, src_rotation.T)

        transform = get_transform_from_rotation_translation(rotation, translation)

        if self.return_corr_indices:                          
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices
        
        # Normal estimation
        src_pcd, ref_pcd = src_points, ref_points
        o3d_src_pcd = to_o3d_pcd(src_pcd)
        o3d_ref_pcd = to_o3d_pcd(ref_pcd)
        o3d_src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        src_normals = np.asarray(o3d_src_pcd.normals).astype(np.float32)
        src_normals = normal_redirect(src_points, src_normals, view_point=self.view_point)
        o3d_ref_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        ref_normals = np.asarray(o3d_ref_pcd.normals).astype(np.float32)
        ref_normals = normal_redirect(ref_points, ref_normals, view_point=self.view_point)

        # pcd0_path_parts, pcd1_path_parts = metadata['pcd0'].split('/'), metadata['pcd1'].split('/')
        # pcd0_last_parts, pcd1_last_parts = pcd0_path_parts[-1].split('_'), pcd1_path_parts[-1].split('_')
        # pcd0_last_parts[0], pcd1_last_parts[0] = 'normal', 'normal'
        # pcd0_path_parts[-1], pcd1_path_parts[-1] = '_'.join(pcd0_last_parts), '_'.join(pcd1_last_parts)
        # pcd0_nomal_path, pcd1_nomal_path = '/'.join(pcd0_path_parts), '/'.join(pcd1_path_parts)
        # os.makedirs(os.path.join('normal1', '/'.join(pcd0_path_parts[:-1])), exist_ok=True)
        # os.makedirs(os.path.join('normal1', '/'.join(pcd1_path_parts[:-1])), exist_ok=True)
        # torch.save(ref_normals, os.path.join('normal1', pcd0_nomal_path))
        # torch.save(src_normals, os.path.join('normal1', pcd1_nomal_path))

        # # visualize(data_save)
        # p0_normals = torch.from_numpy(src_normals).cuda().float()
        # p1_normals = torch.from_numpy(ref_normals).cuda().float()
        # transforms = torch.from_numpy(transform).cuda().float()
        # p0_points, p1_points = torch.from_numpy(src_points).cuda().float(), torch.from_numpy(ref_points).cuda().float()
        # p0_normals = apply_transform(p0_normals, transforms)
        # src, tgt, src_gt = load_points(p0_points, p1_points, transforms)
        
        # src_gt.normals = o3d.utility.Vector3dVector(p0_normals.cpu().numpy())
        # tgt.normals = o3d.utility.Vector3dVector(p1_normals.cpu().numpy())
        # src_gt.paint_uniform_color([0, 1, 0])
        # tgt.paint_uniform_color([1, 0, 0])
        # o3d.visualization.draw_geometries([tgt, src_gt], point_show_normal=False)
        # o3d.visualization.draw_geometries([tgt, src_gt], point_show_normal=True)

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)
        data_dict['ref_normals'] = ref_normals
        data_dict['src_normals'] = src_normals

        return data_dict

#---------------------------
from typing import Union, Tuple

import open3d as o3d
import numpy as np
def to_o3d_pcd(xyz, colors=None, normals=None):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd

def compute_overlap(src: Union[np.ndarray, o3d.geometry.PointCloud],
                    tgt: Union[np.ndarray, o3d.geometry.PointCloud],
                    search_voxel_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes region of overlap between two point clouds.

    Args:
        src: Source point cloud, either a numpy array of shape (N, 3) or
        Open3D PointCloud object
        tgt: Target point cloud similar to src.
        search_voxel_size: Search radius

    Returns:
        has_corr_src: Whether each source point is in the overlap region
        has_corr_tgt: Whether each target point is in the overlap region
        src_tgt_corr: Indices of source to target correspondences
    """

    if isinstance(src, np.ndarray):
        src_pcd = to_o3d_pcd(src)
        src_xyz = src
    else:
        src_pcd = src
        src_xyz = np.asarray(src.points)

    if isinstance(tgt, np.ndarray):
        tgt_pcd = to_o3d_pcd(tgt)
        tgt_xyz = tgt
    else:
        tgt_pcd = tgt
        tgt_xyz = tgt.points

    # Check which points in tgt has a correspondence (i.e. point nearby) in the src,
    # and then in the other direction. As long there's a point nearby, it's
    # considered to be in the overlap region. For correspondences, we require a stronger
    # condition of being mutual matches
    tgt_corr = np.full(tgt_xyz.shape[0], -1)
    pcd_tree = o3d.geometry.KDTreeFlann(src_pcd)
    for i, t in enumerate(tgt_xyz):
        num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(t, search_voxel_size)
        if num_knn > 0:
            tgt_corr[i] = knn_indices[0]
    src_corr = np.full(src_xyz.shape[0], -1)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    for i, s in enumerate(src_xyz):
        num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(s, search_voxel_size)
        if num_knn > 0:
            src_corr[i] = knn_indices[0]

    # Compute mutual correspondences
    src_corr_is_mutual = np.logical_and(tgt_corr[src_corr] == np.arange(len(src_corr)),
                                        src_corr > 0)
    src_tgt_corr = np.stack([np.nonzero(src_corr_is_mutual)[0],
                            src_corr[src_corr_is_mutual]])

    has_corr_src = src_corr >= 0
    has_corr_tgt = tgt_corr >= 0

    return torch.from_numpy(has_corr_src), torch.from_numpy(has_corr_tgt), torch.from_numpy(src_tgt_corr)

def build_dataloader_stack_mode(
    dataset, collate_fn, num_stages, voxel_size, search_radius, # neighbor_limits,
    batch_size=1, num_workers=1, shuffle=False, drop_last=False, distributed=False, precompute_data=True,
):
    pin_memory = False
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = shuffle

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=partial(
            collate_fn,
            num_stages=num_stages,
            voxel_size=voxel_size,
            search_radius=search_radius,
            # neighbor_limits=neighbor_limits,
            precompute_data=precompute_data,
            ),
        worker_init_fn=reset_seed_worker_init_fn, # random.seed(np.random.seed(torch.initial_seed() % (2 ** 32)))
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return data_loader


def registration_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, precompute_data=True
):
    r"""Collate function for registration in stack mode.  在stack模式下配准的整理函数

        Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
        The correspondence indices are within each point cloud without accumulation.

        Args:
            data_dicts (List[Dict])
            num_stages (int)
            voxel_size (float)
            search_radius (float)
            neighbor_limits (List[int])
            precompute_data (bool)

        Returns:
            collated_dict (Dict)
        """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    normals = torch.cat(collated_dict.pop('ref_normals') + collated_dict.pop('src_normals'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)
    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    collated_dict['points'] = points
    collated_dict['normals'] = normals
    collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict



def normal_redirect(points, normals, view_point):
    '''
    Make direction of normals towards the view point
    '''
    vec_dot = np.sum((view_point - points) * normals, axis=-1)
    mask = (vec_dot < 0.)
    redirected_normals = normals.copy()
    redirected_normals[mask] *= -1.
    return redirected_normals





