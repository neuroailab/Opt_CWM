"""Common util functions that can be used throughout the optim-cwm project."""

import os
import random

import einops
import numpy as np
import requests
import torch
import torch.distributed as dist
import tqdm

from utils import constants


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def gen_grid(h_min, h_max, w_min, w_max, len_h, len_w):
    """Generate the coordinate grid.
    Mostly as-is from optimize_counterfactual/notebooks/flow.ipynb"""
    x, y = torch.meshgrid(
        [torch.linspace(w_min, w_max, len_w), torch.linspace(h_min, h_max, len_h)],
        indexing="ij",  # Added this to get rid of the warning
    )
    # Swapping y and x here so that order is row,col not x,y
    grid = torch.stack((y, x), -1).transpose(0, 1).reshape(-1, 2).float()
    return grid


def gather_tensor(tensor: torch.Tensor):
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)

    gather_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, tensor)
    tensor = torch.cat(gather_list)
    return tensor


# NOTE: obj must be picklable
def gather_pyobj(obj):
    gather_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gather_list, obj)
    return gather_list


def size_guard(x, size, message=None):
    """Checks that x has the specified size.

    All dimensions should be specified. Use -1 for placeholders.

    Args:
        x: array/tensor to check size.
        size: tuple of size per dimension.
    """
    if isinstance(x, torch.Tensor):
        x_shape = x.size()
    elif isinstance(x, np.ndarray):
        x_shape = x.shape
    else:
        raise TypeError(f"Unknown type for x: {type(x)}")

    err_msg = f"Expected {size}, got {x_shape}"
    if message is not None:
        err_msg += ". " + message

    assert len(x_shape) == len(size) and all(x_shape[i] == size[i] for i in range(len(size)) if size[i] != -1), err_msg


def rescale_points(pts, og_size, new_size):
    new_pts = torch.zeros_like(pts)
    new_pts[..., 0] = (new_size[0] / og_size[0]) * pts[..., 0]
    new_pts[..., 1] = (new_size[1] / og_size[1]) * pts[..., 1]

    new_pts[..., 0] = torch.clamp(new_pts[..., 0], min=0, max=new_size[0] - 1)
    new_pts[..., 1] = torch.clamp(new_pts[..., 1], min=0, max=new_size[1] - 1)

    return new_pts


def batch_resize_video(video, new_size, mode="bilinear"):
    B, C, T, H, W = video.size()
    grouped = einops.rearrange(video, "b c t h w -> (b t) c h w")
    grouped = torch.nn.functional.interpolate(grouped, new_size, mode=mode)
    resized = einops.rearrange(grouped, "(b t) c h w -> b c t h w", b=B)
    return resized


def compute_2d_argmax(prob_heatmap):
    preds = torch.stack([(x == x.max()).nonzero()[0] for x in prob_heatmap])
    return preds.to(prob_heatmap.device)
    # return (prob_heatmap==torch.max(prob_heatmap)).nonzero()


def video_to_patches(video, patch_t=1, patch_h=8, patch_w=8):
    rearrangement = "b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw) c"
    patches = einops.rearrange(video, rearrangement, pt=patch_t, ph=patch_h, pw=patch_w)

    return patches.view(*patches.shape[:2], -1)


def imagenet_normalize(x, temporal_dim=2):
    """Perform imagenet normalization.

    Subtract mean, then divide by std. dev across color channels.

    Args:
        x: input video tensor: 5D (B, C, T, H, W)
        temporal_dim: Defaults to 2.

    Returns:
        Normalized video
    """
    mean = torch.as_tensor(constants.IMAGENET_DEFAULT_MEAN).to(x.device)[None, None, :, None, None].to(x)
    std = torch.as_tensor(constants.IMAGENET_DEFAULT_STD).to(x.device)[None, None, :, None, None].to(x)
    if temporal_dim == 2:
        mean = mean.transpose(1, 2)
        std = std.transpose(1, 2)
    return (x - mean) / std


def imagenet_unnormalize(x, temporal_dim=2):
    """Undo imagenet normalization."""
    device = x.device
    mean = torch.as_tensor(constants.IMAGENET_DEFAULT_MEAN).to(device)[None, None, :, None, None].to(x)
    std = torch.as_tensor(constants.IMAGENET_DEFAULT_STD).to(device)[None, None, :, None, None].to(x)
    if temporal_dim == 2:
        mean = mean.transpose(1, 2)
        std = std.transpose(1, 2)
    x = x * std + mean
    return x


def extract_attributes_from_maybe_ddp(model, attr):
    """Extract model.attr, handling cases where its either DDP or regular module"""
    if hasattr(model, attr):
        return getattr(model, attr)
    return getattr(model.module, attr)


def download_from_gcloud(url, save_to):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(save_to, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
