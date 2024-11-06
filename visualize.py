import open3d as o3d
import numpy as np
import torch
import os, copy
from modules.ops.transformation import apply_transform

def to_o3d_pcd(pts):
    '''
    From numpy array, make point cloud in open3d format
    :param pts: point cloud (nx3) in numpy array
    :return: pcd: point cloud in open3d format
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
    return pcd

def load_points(p0, p1, transform ):

   
    src_gt = apply_transform(p0, transform)
    src_gt_pcd = to_o3d_pcd(src_gt )
   
    src_pcd = to_o3d_pcd(p0)
    tgt_pcd = to_o3d_pcd(p1)

    return src_pcd, tgt_pcd, src_gt_pcd

def get_correspondences(src_pcd, tgt, transform, dist_dims, dist_threshold=0.04, K=None):
    # src_pcd.transform(trans)
    N, _ = src_pcd.size()
    src_pcd = apply_transform(src_pcd, transform)
    dists = torch.sqrt(pairwise_distance(src_pcd, tgt)) # , normalized=True

    src_min_dist, _ = torch.min(dists, dist_dims)
    gt_score = torch.where(src_min_dist > dist_threshold, 0, 1)
    # gt_score = torch.sigmoid(src_min_dist)
    return gt_score

def pairwise_distance(x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False) -> torch.Tensor:
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    # sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances

def visualize(data_root):
    # data = torch.load(data_root)
    # for i in range(len(datas)):
    data = data_root#[i]
    # p0 = data["p0"]
    # p1 = data["p1"]
    
    p0_f = data["p0_f"]
    p1_f = data["p1_f"]
    p0_f_normal = data["p0_f_normal"]
    p1_f_normal = data["p1_f_normal"]
    p0_corr = data["p0_corr"]
    p1_corr = data["p1_corr"]
    p0_sample = data["p0_sample"]
    p1_sample = data["p1_sample"]
    p0_sample_corr = data["p0_sample_corr"]
    p1_sample_corr = data["p1_sample_corr"]
    p0_center = data["p0_center"]# [0] 
    p1_center = data["p1_center"]# [0]
    transform = data["sample_transform"]  # data["sample_transform"] # 
    gt_transform = data["gt_transform"]  # data["sample_transform"] # 

    p0_points_corr = data["p0_points_corr"]
    p1_points_corr = data["p1_points_corr"]
    p0_points_corr_transform = data["p0_points_corr_transform"]
    p1_points_corr_transform = data["p1_points_corr_transform"]
    p0_points_corr_select = data["p0_points_corr_select"]
    p1_points_corr_select = data["p1_points_corr_select"]
    p0_points_corr_select_normal = data["p0_points_corr_select_normal"]
    p1_points_corr_select_normal = data["p1_points_corr_select_normal"]
    p0_sample_points_corr = data["p0_sample_points_corr"]
    p1_sample_points_corr = data["p1_sample_points_corr"]
    # p0_sample_points_corr_overlap = data["p0_sample_points_corr_overlap"]
    # p1_sample_points_corr_overlap = data["p1_sample_points_corr_overlap"]

    # p0_points_corr_select_normal = apply_transform(p0_points_corr_select_normal.float(), transform)
    
    src_f, tgt_f, src_gt_f = load_points(p0_f, p1_f, transform)
    src_f, _, src_est_f = load_points(p0_f, p1_f, gt_transform)
    # src_c, tgt_c, src_gt_c = load_points(p0_c, p1_c, transform)
    src_sample, tgt_sample, src_gt_sample = load_points(p0_sample, p1_sample, transform)
    
    src_sample_corr, tgt_sample_corr, src_gt_sample_corr = load_points(p0_sample_corr, p1_sample_corr, transform)
    src_corr, tgt_corr, src_gt_corr = load_points(p0_corr, p1_corr, transform)
    src_points_corr, tgt_points_corr, src_points_gt_corr = load_points(p0_points_corr, p1_points_corr, transform)
    src_points_corr_transform, tgt_points_corr_transform, src_points_gt_corr_transform = load_points(p0_points_corr_transform, p1_points_corr_transform, transform)
    src_points_corr_select, tgt_points_corr_select, src_points_gt_corr_select = load_points(p0_points_corr_select, p1_points_corr_select, transform)
    src_sample_points_corr, tgt_sample_points_corr, src_sample_points_gt_corr = load_points(p0_sample_points_corr, p1_sample_points_corr, transform)
    src_center, tgt_center, src_gt_center = load_points(p0_center, p1_center, transform)

    # overlap points num
    # gt0 = get_correspondences(p0_sample_corr, p1_f, transform, dist_dims=-1)
    # gt1 = get_correspondences(p0_f, p1_sample_corr, transform, dist_dims=-2)
    # overlap_num0 = gt0.sum()
    # overlap_num1 = gt1.sum()
    # print(f'src in overlap: {overlap_num0}')
    # print(f'tgt in overlap: {overlap_num1}')


    # src.paint_uniform_color([0, 0.651, 0.929])
    src_est_f.paint_uniform_color([0, 0.651, 0.929])
    src_gt_f.paint_uniform_color([0, 0.651, 0.929])
    src_gt_f.normals = o3d.utility.Vector3dVector(p0_f_normal.cpu().numpy())
    tgt_f.normals = o3d.utility.Vector3dVector(p1_f_normal.cpu().numpy())
    # src_est_f.paint_uniform_color([1, 0.651, 0.929])
    tgt_f.paint_uniform_color([1, 0.706, 0])
    src_gt_corr.paint_uniform_color([0, 1, 0])
    tgt_corr.paint_uniform_color([1, 0, 0])
    
    src_points_gt_corr_select.paint_uniform_color([0.5, 0.651, 0.929])
    tgt_points_corr_select.paint_uniform_color([0.651, 0.5, 0.929])
    src_points_gt_corr_select.normals = o3d.utility.Vector3dVector(p0_points_corr_select_normal.cpu().numpy())
    tgt_points_corr_select.normals = o3d.utility.Vector3dVector(p1_points_corr_select_normal.cpu().numpy())
    src_points_gt_corr.paint_uniform_color([1, 0.651, 0.929])
    tgt_points_corr.paint_uniform_color([0.651, 1, 0.929])
    src_points_gt_corr_transform.paint_uniform_color([0.0, 0.0, 1.0])
    tgt_points_corr_transform.paint_uniform_color([1.0, 1.0, 0])
    src_sample_points_gt_corr.paint_uniform_color([1, 0.929, 0.651])
    tgt_sample_points_corr.paint_uniform_color([0.651, 0.929, 1])
    # src_sample_points_gt_corr_overlap.paint_uniform_color([1, 0.651, 0.929])
    # tgt_sample_points_corr_overlap.paint_uniform_color([0.651, 1, 0.929])

    src_gt_sample.paint_uniform_color([0, 0, 1])
    tgt_sample.paint_uniform_color([0, 0, 0])
    src_gt_sample_corr.paint_uniform_color([0, 0, 1])
    tgt_sample_corr.paint_uniform_color([0, 0, 0])
    src_gt_center.paint_uniform_color([0, 0, 1])
    tgt_center.paint_uniform_color([0, 0, 0])
    
    keys_select = tgt_points_corr_select + src_points_gt_corr_select
    # keys.paint_uniform_color([0, 1, 0])
    corr_keys_select = np.asarray(keys_select.points)
    lines_select = [[j, j + len(p0_points_corr_select)] for j in range(len(p0_points_corr_select))]
    line_set_select = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corr_keys_select),
        lines=o3d.utility.Vector2iVector(lines_select),
    )
    keys_0 = tgt_points_corr + src_points_gt_corr
    keys_0.paint_uniform_color([1, 0, 0])
    corr_keys_0 = np.asarray(keys_0.points)
    lines_0 = [[j, j + len(p0_points_corr)] for j in range(len(p0_points_corr))]
    line_set_0 = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corr_keys_0),
        lines=o3d.utility.Vector2iVector(lines_0),
    )
    keys_sample = tgt_sample_points_corr + src_sample_points_gt_corr
    keys_sample.paint_uniform_color([1, 0, 0])
    corr_keys_sample = np.asarray(keys_sample.points)
    lines_sample = [[j, j + len(p0_sample_points_corr)] for j in range(len(p0_sample_points_corr))]
    line_set_sample = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corr_keys_sample),
        lines=o3d.utility.Vector2iVector(lines_sample),
    )
    # src_points_gt_corr_transform.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.01, 30))
    o3d.visualization.draw([{
        "name": "tgt_f",
        "geometry": tgt_f
    }, {
        "name": "src_f",
        "geometry": src_gt_f
    }, {
        "name": "src_gt_f",
        "geometry": src_est_f
    }, {
        "name": "tgt_corr",
        "geometry": tgt_corr
    }, {
        "name": "src_corr",
        "geometry": src_gt_corr
    },{
        "name": "tgt_points_corr_select",
        "geometry": tgt_points_corr_select
    }, {
        "name": "src_points_corr_select",
        "geometry": src_points_gt_corr_select
    },{
        "name": "inlier_line_select",          
        "geometry": line_set_select
    },{
        "name": "tgt_points_corr_transform",
        "geometry": tgt_points_corr_transform
    }, {
        "name": "src_points_corr_transform",
        "geometry": src_points_gt_corr_transform
    },{
        "name": "tgt_points_corr",
        "geometry": tgt_points_corr
    }, {
        "name": "src_points_corr",
        "geometry": src_points_gt_corr
    # },{
    #     "name": "inlier_line_corr",         
    #     "geometry": line_set_0
    },], show_ui=True,)

    print('-----------')


def cluster(data_root):
    # data = torch.load(data_root)
    data = data_root
    
    p0_m = data["p0_f"]
    p1_m = data["p1_f"]
    p0_c = data["p0_c"]
    p1_c = data["p1_c"]
    p0_corr = data["p0_corr"][0]
    p1_corr = data["p1_corr"][0]
    transform = data["gt_transform"]

    import numpy as np
    from sklearn.cluster import DBSCAN

    # points = np.loadtxt('point_cloud.txt')
    points = p1_corr.cpu().numpy()

    epsilon = 0.5  
    min_samples = 10  
    cluster_count = 0  

    while True:
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        labels = dbscan.fit_predict(points)

        num_clusters = len(np.unique(labels))
        if num_clusters > cluster_count:
            cluster_count = num_clusters
        else:
            break

        epsilon *= 1.1
        min_samples += 1


    points_cluster = []
    for i in np.unique(labels): 
        indices = np.where(labels==i)[0]
        points_cluster.append(torch.tensor(points[indices]))

    
    points_0 = to_o3d_pcd(points_cluster[0])
    p1_f = to_o3d_pcd(p1_f)
    p1_f.paint_uniform_color([1, 0.706, 0])
    points_0.paint_uniform_color([0, 1, 0])
    if num_clusters > 1:
        points_1 = to_o3d_pcd(points_cluster[1])
        points_1.paint_uniform_color([0, 0, 1])
    else:
        points_1 = points_0
    if num_clusters > 2:
        points_2 = to_o3d_pcd(points_cluster[2])
        points_2.paint_uniform_color([1, 0, 1])
    else:
        points_2 = points_0
    o3d.visualization.draw([{
        "name": "p1_m",
        "geometry": p1_m
    }, {
        "name": "points_0",
        "geometry": points_0
    }, {
        "name": "points_1",
        "geometry": points_1
    }, {
        "name": "points_2",
        "geometry": points_2
    }, 
    
    ], show_ui=True)


if __name__ == "__main__":
    data_root = f'output/data.pth'
    cluster(data_root)
