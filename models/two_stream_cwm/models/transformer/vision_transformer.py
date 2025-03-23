import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.two_stream_cwm import ThreeTuple
from models.two_stream_cwm.models.transformer.params import InputParams, ViTParams
from models.two_stream_cwm.models.transformer.utils import UsesTypicalInit


class VisionTransformer(UsesTypicalInit, nn.Module):
    def __init__(self, params: ViTParams):
        super().__init__()

        self.num_classes = params.num_classes
        self.embed_dim = params.embed_dim

        # stochastic depth decay rule
        drop_path_rates = [x.item() for x in torch.linspace(0, params.drop_path_rate, params.depth)]
        self.blocks = nn.ModuleList(
            [
                params.block_func(
                    drop_path=drop_path_rates[i],
                    params=params.block_params,
                    **params.block_func_kwargs,
                )
                for i in range(params.depth)
            ]
        )
        self.norm = params.norm_layer(params.embed_dim)
        self.head = nn.Linear(params.embed_dim, params.num_classes) if params.num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def forward_block(self, x: Tensor, idx: int) -> Tensor:
        """Run only blocks[`idx`] on the given input

        Args:
            x: input to process
            idx: block index (0-indexed)
        """
        return self.blocks[idx](x)


class VideoPatchEmbed(nn.Module):
    """Video to Patch Embedding"""

    def __init__(
        self,
        params: InputParams,
        patch_size: ThreeTuple[int] = (2, 16, 16),
        embed_dim: int = 768,
    ):
        """Patch embedding for videos"""
        super().__init__()

        self.in_chans = params.in_chans
        self.inp_size = params.inp_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.num_frames = int(params.num_frames)

        # total number of patches depends on how many patches we can fit in each
        # of the three dims (time, height, width)
        self.num_patches = math.prod([self.inp_size[dim] // self.patch_size[dim] for dim in range(3)])

        # 3D conv with kernel size == stride to make even patches
        self.proj = nn.Conv3d(
            in_channels=params.in_chans,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Channels, Time, Height, Width)
        """
        B, C, T, H, W = x.shape
        assert (T % self.patch_size[0] == 0) and (H % self.patch_size[1] == 0) and (W % self.patch_size[2] == 0), (
            f"Input size ({T}*{H}*{W}) doesn't match model "
            f"({self.inp_size[0]}*{self.inp_size[1]}*{self.inp_size[2]})."
        )

        # Conv3D isn't implemented in mps, so loop a bunch of 2D convs
        if x.device.type == "mps":
            t, h, w = self.proj.weight.shape[-3:]
            num_time_patches = T // t
            x_slices = [_x.reshape(B, C * t, H, W) for _x in torch.split(x, [t] * num_time_patches, dim=2)]

            # reshape weight matrix
            # From: (embed_dim, in_chan, tubelet_size, patch_height, patch_width)
            # To: (embed_dim, in_chan * tubelet_size, patch_height, patch_width)
            weights = self.proj.weight.view(-1, self.proj.weight.size(1) * t, h, w)
            x_out = [F.conv2d(_x, weight=weights, bias=self.proj.bias, stride=(h, w)) for _x in x_slices]
            x = torch.stack(x_out, 2).flatten(2).transpose(1, 2)
        else:
            x = self.proj(x).flatten(2).transpose(1, 2)
        return x
