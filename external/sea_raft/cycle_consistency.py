""" 
These functions are from 
https://github.com/neuroailab/cwm_dynamics/blob/main/cwm/inference/flow/flow_utils.py
with only minor modifications, if any.
"""

import torch

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn import functional as F
from torchvision import transforms


def sampling_grid(height, width):
    H, W = height, width
    grid = torch.stack([torch.arange(W).view(1, -1).repeat(H, 1), torch.arange(H).view(-1, 1).repeat(1, W)], -1)
    grid = grid.view(1, H, W, 2)
    return grid


def normalize_sampling_grid(coords):
    assert len(coords.shape) == 4, coords.shape
    assert coords.size(-1) == 2, coords.shape
    H, W = coords.shape[-3:-1]
    xs, ys = coords.split([1, 1], -1)
    xs = 2 * xs / (W - 1) - 1
    ys = 2 * ys / (H - 1) - 1
    return torch.cat([xs, ys], -1)


def backward_warp(img2, flow, do_mask=False):
    """
    Grid sample from img2 using the flow from img1->img2 to get a prediction of img1.

    flow: [B,2,H',W'] in units of pixels at its current resolution. The two channels
          should be (x,y) where larger y values correspond to lower parts of the image.
    """

    ## resize the flow to the image size.
    ## since flow has units of pixels, its values need to be rescaled accordingly.
    if list(img2.shape[-2:]) != list(flow.shape[-2:]):
        scale = [img2.size(-1) / flow.size(-1), img2.size(-2) / flow.size(-2)]  # x  # y
        scale = torch.tensor(scale).view(1, 2, 1, 1).to(flow.device)
        flow = scale * transforms.Resize(img2.shape[-2:])(flow)  # defaults to bilinear

    B, C, H, W = img2.shape

    ## use flow to warp sampling grid
    grid = sampling_grid(H, W).to(flow.device) + flow.permute(0, 2, 3, 1)

    ## put grid in normalized image coordinates
    grid = normalize_sampling_grid(grid)

    ## backward warp, i.e. sample pixel (x,y) from (x+flow_x, y+flow_y)
    img1_pred = F.grid_sample(img2, grid, align_corners=True)

    if do_mask:
        mask = (grid[..., 0] > -1) & (grid[..., 0] < 1) & (grid[..., 1] > -1) & (grid[..., 1] < 1)
        mask = mask[:, None].to(img2.dtype)
        return (img1_pred, mask)

    else:
        return (img1_pred, torch.ones_like(grid[..., 0][:, None]).float())


def l2_norm(x):
    return x.square().sum(-3, True).sqrt()


def get_occ_masks(flow_fwd, flow_bck, occ_thresh=0.5, how="rel", return_diffs=False):
    fwd_bck_cycle, _ = backward_warp(img2=flow_bck, flow=flow_fwd)
    flow_diff_fwd = flow_fwd + fwd_bck_cycle

    bck_fwd_cycle, _ = backward_warp(img2=flow_fwd, flow=flow_bck)
    flow_diff_bck = flow_bck + bck_fwd_cycle

    if how == "rel":
        # This was the function as previously written

        norm_fwd = l2_norm(flow_fwd) ** 2 + l2_norm(fwd_bck_cycle) ** 2
        norm_bck = l2_norm(flow_bck) ** 2 + l2_norm(bck_fwd_cycle) ** 2

        occ_thresh_fwd = occ_thresh * norm_fwd + 0.5
        occ_thresh_bck = occ_thresh * norm_bck + 0.5

        occ_mask_fwd = 1 - (l2_norm(flow_diff_fwd) ** 2 > occ_thresh_fwd).float()
        occ_mask_bck = 1 - (l2_norm(flow_diff_bck) ** 2 > occ_thresh_bck).float()

    elif how == "abs":
        # Modified to use absolute pixel threshold
        occ_mask_fwd = 1 - (l2_norm(flow_diff_fwd) > occ_thresh).float()
        occ_mask_bck = 1 - (l2_norm(flow_diff_bck) > occ_thresh).float()

    else:
        raise ValueError(f"Invalid thresholding method: {how=}")

    if return_diffs:
        return occ_mask_fwd, occ_mask_bck, flow_diff_fwd, flow_diff_bck
    return occ_mask_fwd, occ_mask_bck
