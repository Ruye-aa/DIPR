# from geotransformer.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
import os
import random
from functools import partial
import numpy as np
import pickle
import torch.utils.data
from typing import Dict

from utils.common import load_pickle
from utils.pointcloud import random_sample_rotation_v2, get_transform_from_rotation_translation, random_sample_rotation
from utils.registration import get_correspondences
from utils.torch import reset_seed_worker_init_fn
from modules.ops import grid_subsample, radius_search
from modules.ops.transformation import apply_transform

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

    neighbor_limits = calibrate_neighbors_stack_mode(      
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    train_loader = build_dataloader_stack_mode(             
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
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
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
    )
    return train_loader, valid_loader, neighbor_limits

def test_data_loader(cfg, benchmark):
    train_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(      
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

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
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, neighbor_limits


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

    def __len__(self):
        return len(self.metadata_list)

    def _load_point_cloud(self, file_name):
        points = torch.load(os.path.join(self.data_root, file_name))
        # NOTE: setting "point_limit" with "num_workers" > 1 will cause nondeterminism.
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indeces = np.random.permutation(points.shape[0])[:self.point_limit]  
            points = points[indeces]
        return points

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
        ref_points = self._load_point_cloud(metadata['pcd0']) 
        src_points = self._load_point_cloud(metadata['pcd1'])

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
        
        src_overlap_mask, ref_overlap_mask, src_tgt_corr = compute_overlap(
                apply_transform(torch.from_numpy(src_points), torch.from_numpy(transform)).cpu().numpy(),
                ref_points,
                self.search_voxel_size,
            )

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)
        data_dict['ref_overlap'] = ref_overlap_mask
        data_dict['src_overlap'] = src_overlap_mask

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



def calibrate_neighbors_stack_mode(                     
    dataset, collate_fn, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=2000
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))  
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages # [180, 180, 180, 180]

    # Get histogram of neighborhood sizes i in 1 epoch max. 
    for i in range(len(dataset)):                       
        data_dict = collate_fn(
            [dataset[i]], num_stages, voxel_size, search_radius, max_neighbor_limits, precompute_data=True
        )                                              

        # update histogram
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits

def build_dataloader_stack_mode(
    dataset, collate_fn, num_stages, voxel_size, search_radius, neighbor_limits,
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
            neighbor_limits=neighbor_limits,
            precompute_data=precompute_data,
            ),
        worker_init_fn=reset_seed_worker_init_fn, # random.seed(np.random.seed(torch.initial_seed() % (2 ** 32)))
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return data_loader


def registration_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for registration in stack mode. 

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
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)

    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict

def precompute_data_stack_mode(points, lengths, num_stages, voxel_size, radius, neighbor_limits):
    assert num_stages == len(neighbor_limits)

    points_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    # grid subsampling
    for i in range(num_stages):
        if i > 0:
            points, lengths = grid_subsample(points, lengths, voxel_size=voxel_size)
        points_list.append(points)
        lengths_list.append(lengths)
        voxel_size *= 2

    # radius search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        neighbors = radius_search(
            cur_points,
            cur_points,
            cur_lengths,
            cur_lengths,
            radius,
            neighbor_limits[i],
        )
        neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                radius,
                neighbor_limits[i],
            )
            subsampling_list.append(subsampling)

            upsampling = radius_search(
                cur_points,
                sub_points,
                cur_lengths,
                sub_lengths,
                radius * 2,
                neighbor_limits[i + 1],
            )
            upsampling_list.append(upsampling)

        radius *= 2

    return {
        'points': points_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
    }



