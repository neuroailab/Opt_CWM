from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import torch.nn as nn

from models.two_stream_cwm import Kwargs, ThreeTuple
from models.two_stream_cwm.models.transformer.transformer import Block, BlockParams

# type alias
Module = nn.modules.module.Module


@dataclass
class InputParams:
    inp_size: ThreeTuple[int] = (2, 224, 224)
    in_chans: int = 3

    @property
    def num_frames(self) -> int:
        return self.inp_size[0]


@dataclass
class ViTParams:
    """Parameters shared by the encoder and decoder. They need not share the exact same
    values, but the fields are identical
    """

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


@dataclass
class EncoderParams(ViTParams):
    inp: InputParams = field(default_factory=lambda: InputParams())
    patch_size: ThreeTuple[int] = (1, 16, 16)
