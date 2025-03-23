""" 
These functions are from 
https://github.com/neuroailab/cwm_dynamics/blob/main/cwm/inference/flow/flow_utils.py
with only minor modifications, if any.
"""

import torch

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn import functional as F
from torchvision import transforms
import numpy as np


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


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, "input flow must have three dimensions"
    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)
