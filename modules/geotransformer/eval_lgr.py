import torch
import numpy as np
from modules.ops import apply_transform
from modules.registration import WeightedProcrustes


def convert_to_batch(ref_corr_points, src_corr_points, corr_scores, chunks):
    r"""Convert stacked correspondences to batched points.

    The extracted dense correspondences from all patch correspondences are stacked. However, to compute the
    transformations from all patch correspondences in parallel, the dense correspondences need to be reorganized
    into a batch.

    Args:
        ref_corr_points (Tensor): (C, 3)
        src_corr_points (Tensor): (C, 3)
        corr_scores (Tensor): (C,)
        chunks (List[Tuple[int, int]]): the starting index and ending index of each patch correspondences.

    Returns:
        batch_ref_corr_points (Tensor): (B, K, 3), padded with zeros.
        batch_src_corr_points (Tensor): (B, K, 3), padded with zeros.
        batch_corr_scores (Tensor): (B, K), padded with zeros.
    """
    batch_size = len(chunks)
    indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
    ref_corr_points = ref_corr_points[indices]  # (total, 3)
    src_corr_points = src_corr_points[indices]  # (total, 3)
    corr_scores = corr_scores[indices]  # (total,)

    max_corr = np.max([y - x for x, y in chunks])
    target_chunks = [(i * max_corr, i * max_corr + y - x) for i, (x, y) in enumerate(chunks)]
    indices = torch.cat([torch.arange(x, y) for x, y in target_chunks], dim=0).cuda()
    indices0 = indices.unsqueeze(1).expand(indices.shape[0], 3)  # (total,) -> (total, 3)
    indices1 = torch.arange(3).unsqueeze(0).expand(indices.shape[0], 3).cuda()  # (3,) -> (total, 3)

    batch_ref_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
    batch_ref_corr_points.index_put_([indices0, indices1], ref_corr_points)
    batch_ref_corr_points = batch_ref_corr_points.view(batch_size, max_corr, 3)

    batch_src_corr_points = torch.zeros(batch_size * max_corr, 3).cuda()
    batch_src_corr_points.index_put_([indices0, indices1], src_corr_points)
    batch_src_corr_points = batch_src_corr_points.view(batch_size, max_corr, 3)

    batch_corr_scores = torch.zeros(batch_size * max_corr).cuda()
    batch_corr_scores.index_put_([indices], corr_scores)
    batch_corr_scores = batch_corr_scores.view(batch_size, max_corr)

    return batch_ref_corr_points, batch_src_corr_points, batch_corr_scores

def recompute_correspondence_scores(ref_corr_points, src_corr_points, corr_scores, estimated_transform, acceptance_radius = 1.0):
    aligned_src_corr_points = apply_transform(src_corr_points, estimated_transform)
    corr_residuals = torch.linalg.norm(ref_corr_points - aligned_src_corr_points, dim=1)
    inlier_masks = torch.lt(corr_residuals, acceptance_radius)
    new_corr_scores = corr_scores * inlier_masks.float()
    return new_corr_scores

def local_to_global_registration(global_ref_corr_points, global_src_corr_points, global_corr_scores, batch_indices):
    
    procrustes = WeightedProcrustes(return_transform=True)

    correspondence_limit = None
    correspondence_threshold = 3
    acceptance_radius =  0.1 # for 3d/3dlo; 1.0 for kitty
    num_refinement_steps = 5

    # build verification set
    if correspondence_limit is not None and global_corr_scores.shape[0] > correspondence_limit:
        corr_scores, sel_indices = global_corr_scores.topk(k=correspondence_limit, largest=True)
        ref_corr_points = global_ref_corr_points[sel_indices]
        src_corr_points = global_src_corr_points[sel_indices]
    else:
        ref_corr_points = global_ref_corr_points
        src_corr_points = global_src_corr_points
        corr_scores = global_corr_scores

    # compute starting and ending index of each patch correspondence.
    # torch.nonzero is row-major, so the correspondences from the same patch correspondence are consecutive.
    # find the first occurrence of each batch index, then the chunk of this batch can be obtained.
    unique_masks = torch.ne(batch_indices[1:], batch_indices[:-1])
    unique_indices = torch.nonzero(unique_masks, as_tuple=True)[0] + 1
    unique_indices = unique_indices.detach().cpu().numpy().tolist()
    unique_indices = [0] + unique_indices + [batch_indices.shape[0]]
    chunks = [
        (x, y) for x, y in zip(unique_indices[:-1], unique_indices[1:]) if y - x >= correspondence_threshold
    ]

    batch_size = len(chunks)
    if batch_size > 0:
        # local registration
        batch_ref_corr_points, batch_src_corr_points, batch_corr_scores = convert_to_batch(
            global_ref_corr_points, global_src_corr_points, global_corr_scores, chunks
        )
        batch_transforms = procrustes(batch_src_corr_points, batch_ref_corr_points, batch_corr_scores)
        batch_aligned_src_corr_points = apply_transform(src_corr_points.unsqueeze(0), batch_transforms)
        batch_corr_residuals = torch.linalg.norm(
            ref_corr_points.unsqueeze(0) - batch_aligned_src_corr_points, dim=2
        )
        batch_inlier_masks = torch.lt(batch_corr_residuals, acceptance_radius)  # (P, N)
        best_index = batch_inlier_masks.sum(dim=1).argmax()
        cur_corr_scores = corr_scores * batch_inlier_masks[best_index].float()
    else:
        # degenerate: initialize transformation with all correspondences
        estimated_transform = procrustes(src_corr_points, ref_corr_points, corr_scores)
        cur_corr_scores = recompute_correspondence_scores(
            ref_corr_points, src_corr_points, corr_scores, estimated_transform, acceptance_radius=acceptance_radius
        )

    # global refinement
    estimated_transform = procrustes(src_corr_points, ref_corr_points, cur_corr_scores)
    for _ in range(num_refinement_steps - 1):
        cur_corr_scores = recompute_correspondence_scores(
            ref_corr_points, src_corr_points, corr_scores, estimated_transform, acceptance_radius=acceptance_radius
        )
        estimated_transform = procrustes(src_corr_points, ref_corr_points, cur_corr_scores)

    return estimated_transform
