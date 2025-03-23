from dataclasses import dataclass

from models.two_stream_cwm import ThreeTuple
from models.two_stream_cwm.models.transformer.params import InputParams
from models.two_stream_cwm.models.two_stream_vmae.two_stream.two_stream import TwoStreamTransformerParams


@dataclass
class StreamParams:
    inp: InputParams
    embed_dim: int
    patch_size: ThreeTuple[int]


@dataclass
class TwoStreamEncoderParams:
    transformer_params: TwoStreamTransformerParams
    primary: StreamParams
    secondary: StreamParams
    use_learnable_pos_emb: bool = False

    def __post_init__(self):
        """Enforce that embed dims match transformer block dims"""
        self.transformer_params.block_params.primary.dim = self.primary.embed_dim
        self.transformer_params.block_params.secondary.dim = self.secondary.embed_dim
