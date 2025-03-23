import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.two_stream_cwm.models.transformer.utils import is_flash_attention_available


class OneWayCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        downsample_factor: int = 1,
        attn_drop_rate: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Transformer Cross Attention layer

        modified from Segment Anything (Meta) Attention
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.inner_dim = dim // downsample_factor
        self.head_dim = self.inner_dim // self.num_heads
        assert (
            self.inner_dim % num_heads == 0
        ), "inner dimensionality must be evenly divisible by number of attention heads"

        self.scale = qk_scale or self.head_dim**-0.5

        self.use_flash_attention = is_flash_attention_available()
        assert self.use_flash_attention, "Flash attention not used in OneWayCrossAttention"
        # print("\n\n\nUsing flash attention in cwm.models.two_stream.OneWayCrossAttention")

        # submodules
        self.qkv = nn.ModuleDict(
            {
                "q": nn.Linear(self.dim, self.inner_dim, bias=qkv_bias),
                "k": nn.Linear(self.dim, self.inner_dim, bias=qkv_bias),
                "v": nn.Linear(self.dim, self.inner_dim, bias=qkv_bias),
            }
        )
        self.proj = nn.Linear(self.inner_dim, self.dim)

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj_drop = nn.Dropout(proj_drop)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        B, N, C = x.shape
        x = x.reshape(B, N, num_heads, C // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        B, N_heads, N_tokens, D = x.shape
        x = x.transpose(1, 2)
        return x.reshape(B, N_tokens, N_heads * D)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        B, _, C = q.shape

        # input projections
        q = self.qkv["q"](q)
        k = self.qkv["k"](k)
        v = self.qkv["v"](v)

        # attention: out has shape (B, Nk, C)
        if self.use_flash_attention:
            out = F.scaled_dot_product_attention(
                query=q, key=k, value=v, dropout_p=self.attn_drop.p, attn_mask=attn_mask
            )
        else:
            q = q * self.scale
            q = self._separate_heads(q, self.num_heads)
            k = self._separate_heads(k, self.num_heads)
            v = self._separate_heads(v, self.num_heads)

            q.size(-1)
            attn = q @ k.permute(0, 1, 3, 2)  # (B, num_heads, Nq, Nk)
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask == 1, float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)

            out = attn @ v
            out = self._recombine_heads(out)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out
