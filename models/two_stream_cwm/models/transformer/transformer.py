from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.two_stream_cwm import Kwargs
from models.two_stream_cwm.models.layers.drop import DropPath
from models.two_stream_cwm.models.layers.mlp import MLP
from models.two_stream_cwm.models.transformer.utils import is_flash_attention_available


@dataclass
class AttentionParams:
    num_heads: int = 8
    qkv_bias: bool = False
    qk_scale: float | None = None
    attn_drop_rate: float = 0.0


@dataclass
class BlockParams:
    """Parameters for each block."""

    attn: AttentionParams = field(default_factory=lambda: AttentionParams())

    dim: int = 768
    mlp_ratio: float = 4.0
    downsample_factor: int = 1
    drop_rate: float = 0.0
    act_layer: Type[nn.Module] = nn.GELU
    norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6)
    init_values: float | None = None

    @property
    def hidden_dim(self) -> int:
        return int(self.dim * self.mlp_ratio)

    @property
    def inner_dim(self) -> int:
        return self.dim // self.downsample_factor


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads=8,
        qkv_bias=False,
        qk_scale: float | None = None,
        attn_drop_rate=0.0,
        proj_drop=0.0,
    ):
        """Transformer Attention layer

        Args:
            dim: total dimensionality across all heads
            num_heads: number of attention heads. Defaults to 8.
            qkv_bias: whether or not to use bias in the linear QKV op.
                Defaults to False.
            qk_scale: If not None, the value to scale Q by. Defaults to None.
            attn_drop_rate: _description_. Defaults to 0.0.
            proj_drop: _description_. Defaults to 0.0.
            attn_head_dim: _description_. Defaults to None.
            flash_attention: _description_. Defaults to False.

        NB:
            In the VideoMAE implementation, they have a lot of conditional branching
            to avoid using bias or scaling for the `k` term, but the math works out
            so that it doesn't matter (and I doubt it makes performance better) as long
            as a softmax layer is used after -- and it always is for attn layers.
            See https://github.com/microsoft/unilm/issues/510.
            For those reasons, `timm` and others (including us, here) don't bother with
            those gymnastics
        """
        super().__init__()
        assert dim % num_heads == 0, "total dimensionality must be evenly divisible by number of attention heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.use_flash_attention = is_flash_attention_available()
        assert self.use_flash_attention, "Flash attention not used in Attention"
        # print("\n\n\nUsing flash attention in cwm.models.transformer.Attention")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_mask=None):
        B, N, C = x.shape

        # apply QKV to the input, going from (B, N, C) to (B, N, C * 3)
        qkv = self.qkv(x)

        # then reshape to (B, N, num_heads, dim_per_head)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)

        # finally, permute axes to (3, B, num_heads, N, dim_per_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(dim=0)

        if self.use_flash_attention:
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, dropout_p=self.attn_drop.p, attn_mask=attn_mask)
        else:
            # scale query tensor
            q = q * self.scale

            # flip last 2 dimensions of k, so that it's (B, num_heads, dim_per_head, N),
            # then multiply against q. Matmul will be :
            # (B, num_heads, N, dim_per_head) x (B, num_heads, dim_per_head, N) and the
            # result will be (B, num_heads, N, N), i.e., the element-to-element attn
            attn = q @ k.transpose(-2, -1)

            # mask out attn if we were provided a mask
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask == 1, float("-inf"))

            # softmax and dropout
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # finally, compute x by multiplying attn:
            # (B, NH, N, N) x (B, NH, N, dim_per_head) = (B, NH, N, N)
            x = attn @ v

        # flip back to (B, N, num_heads, N) then merge the last two dimensions to get
        # back to (B, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)

        # final projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        params: BlockParams,
        drop_path: float = 0.0,
    ):
        """Vision transformer block

        Args:
            drop_path: DropPath probability. Defaults to 0.0.
        """
        super().__init__()
        self.params = params
        self.norm1 = params.norm_layer(params.dim)
        self.attn = Attention(
            params.dim,
            proj_drop=params.drop_rate,
            num_heads=params.attn.num_heads,
            qkv_bias=params.attn.qkv_bias,
            qk_scale=params.attn.qk_scale,
            attn_drop_rate=params.attn.attn_drop_rate,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = params.norm_layer(params.dim)

        self.mlp = MLP(
            in_features=params.dim,
            hidden_features=params.hidden_dim,
            act_layer=params.act_layer,
            drop_rate=params.drop_rate,
        )

        init_provided = params.init_values is not None and params.init_values > 0
        self.gamma_1: nn.Parameter | None = (
            nn.Parameter(params.init_values * torch.ones((params.dim))) if init_provided else None
        )
        self.gamma_2: nn.Parameter | None = (
            nn.Parameter(params.init_values * torch.ones((params.dim))) if init_provided else None
        )

    def forward(self, x, attn_mask=None):
        if self.gamma_1 is None or self.gamma_2 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), attn_mask=attn_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


@dataclass
class TransformerParams:
    """No input shape or patch size"""

    block_params: BlockParams
    block_func: type[nn.Module] = Block
    block_func_kwargs: Kwargs = field(default_factory=dict)

    norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6)
    embed_dim: int = 768
    num_classes: int = 0
    depth: int = 12
    num_heads: int = 12
    drop_path_rate: float = 0.0
    use_learnable_pos_emb: bool = False
    apply_positional_embedding_all_layers: bool = False

    def __post_init__(self):
        self.block_params.dim = self.embed_dim
