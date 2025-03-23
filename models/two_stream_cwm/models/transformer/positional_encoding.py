from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from models.two_stream_cwm.models.layers.mlp import MLP, MLPParams
from models.two_stream_cwm.models.transformer.utils import expand_pos_embed


def sinusoidal_pos_embedding(
    positions: int | list[int] | Tensor,
    hidden_dim: int,
    dtype=torch.float32,
    device="cpu",
) -> Tensor:
    """Generate a sinusoidal positional embedding.

    Args:
        positions: If an integer, specifies the number of positions to encode. If
            iterable, this is interpreted as the values of those positions directly.
        hidden_dim: The dimensionality of the positional embedding. For a transformer,
            this should be the dimensionality of the tokens that the PE will be added to
        dtype: datatype to cast the embedding to. Defaults to torch.float32.
        device: device to place the embedding on. Defaults to "cpu".

    Returns:
        A tensor of shape (len(positions), hidden_dim)
    """
    # parse type of position input, and ensure it's iterable
    if isinstance(positions, int):
        positions = torch.arange(positions, requires_grad=True, dtype=dtype, device=device)
    elif isinstance(positions, Tensor):
        positions = positions.to(dtype).to(device)
    else:
        assert hasattr(positions, "__len__"), "positions is not iterable"
        positions = torch.tensor(positions, dtype=dtype).to(device)

    # frequencies are a linear count up to the number of dimensions required
    freqs = torch.arange(hidden_dim).to(dtype).to(device)

    # add an empty batch dimension if positions require
    if len(positions.shape) == 2:
        freqs = freqs.unsqueeze(dim=0)

    # scale frequencies
    freqs = torch.pow(10000, 2 * (torch.div(freqs, 2, rounding_mode="trunc")) / hidden_dim)

    out = positions.unsqueeze(dim=-1) / freqs.unsqueeze(dim=-2)

    # compute sin and cosine
    out_sin = torch.sin(out[:, 0::2])
    out_cos = torch.cos(out[:, 1::2])

    # stack and return as (batch, len(positions), dim)
    return torch.stack([out_sin, out_cos], -1).view(-1, positions.size(-1), hidden_dim)


class PositionalEncoding(nn.Module, metaclass=ABCMeta):
    """
    Base class for predicting a positional embedding for
    a predictor with a given patch size and embedding dimension
    """

    def __init__(
        self,
        dim: int,
        patch_size=(1, 8, 8),
        input_size=(2, 224, 224),
        embed_dim=768,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size
        self.pt, self.ph, self.pw = patch_size
        self.input_size = input_size
        self.embed_dim = embed_dim

    @property
    def mask_size(self):
        T, H, W = self.input_size
        return T // self.pt, H // self.pw, W // self.ph

    @property
    def num_patches(self):
        return np.prod(self.mask_size)

    @property
    def num_frames(self):
        return self.mask_size[0]

    @property
    def num_patches_per_frame(self):
        return self.mask_size[-2] * self.mask_size[-1]

    @abstractmethod
    def encode_as_predictor_positional_embedding(self, z: Tensor) -> tuple[Tensor, Tensor]:
        pass

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Convert a set of tokens of dimension self.dim
        to dimension self.embed_dim.

        This may depend on the video input size and patch size of
        the masked predictor that will use these embeddings.
        """
        return self.encode_as_predictor_positional_embedding(z)


class WeightedSinusoidalPositionalEncoding(PositionalEncoding):
    """
    Output the softmax-weighted sinusoidal_positional_encoding
    that is standard in the VMAE.

    Must have input dimension equal to number of patches to weight over.

    If the inputs do not, learn an MLP to embed in this dimension.
    """

    def __init__(
        self,
        *args,
        mlp_params: MLPParams | None = None,
        allowed_frames: tuple[int, ...] | int | None = (1,),
        embed_dim: int,
        decoder_embed_dim: int,
        **kwargs,
    ) -> None:
        """
        If mlp_params is None, just embed as a linear layer
        (if self.dim != self.num_patches) else nn.Identity

        allowed_frames: controls which frames the predicted embeddings
        can belong to. For typical 2-frame predictor, this will be (1,).
        """
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        if allowed_frames is None:
            self.allowed_frames = tuple(range(self.num_frames))
        elif isinstance(allowed_frames, int):
            self.allowed_frames = (allowed_frames,)
        else:
            self.allowed_frames = allowed_frames

        # create the positional embedding
        self.pos_embed = self._init_pos_embed(dim=self.embed_dim)
        self.decoder_pos_embed = self._init_pos_embed(dim=self.decoder_embed_dim)

        # create the linear or mlp layer to the right number of patches
        self.allowed_num_patches = self.pos_embed.size(1)

        # specify the type explicitly to make mypy happy
        self.mlp_to_patches: MLP | nn.Linear | nn.Identity
        if mlp_params is not None:
            self.mlp_to_patches = MLP(
                in_features=self.dim,
                hidden_features=mlp_params.hidden_features,
                out_features=self.allowed_num_patches,
                act_layer=mlp_params.act_layer,
                bias=mlp_params.bias,
                drop_rate=0,
            )
        elif self.allowed_num_patches != self.dim:
            self.mlp_to_patches = nn.Linear(in_features=self.dim, out_features=self.allowed_num_patches, bias=False)
        else:
            self.mlp_to_patches = nn.Identity()

    def _init_pos_embed(self, dim) -> Tensor:
        pos_embed = sinusoidal_pos_embedding(
            positions=int(self.num_patches), hidden_dim=dim
        )  # (1, num_patches, embed_dim)

        # only allowed frame positions
        pos_embed = pos_embed.view(1, self.num_frames, self.num_patches_per_frame, dim)
        pos_embed = torch.stack([pos_embed[:, frame] for frame in self.allowed_frames], dim=1)
        pos_embed = pos_embed.view(1, -1, dim)
        return pos_embed

    def encode_as_predictor_positional_embedding(self, z: Tensor) -> tuple[Tensor, Tensor]:
        # 6-dim -> 784-dim (number of total positions)
        z = self.mlp_to_patches(z)

        # softmax-weighted average over positions
        z = torch.softmax(z, -1)

        # expand pos embed
        pos_embed = expand_pos_embed(self.pos_embed, expand_as=z)
        decoder_pos_embed = expand_pos_embed(self.decoder_pos_embed, expand_as=z)

        # compute weighted sums of encoder and decoder pos embed
        weighted_encoder_pos_embed = torch.einsum("bnp,bpd->bnd", z, pos_embed)
        weighted_decoder_pos_embed = torch.einsum("bnp,bpd->bnd", z, decoder_pos_embed)
        return weighted_encoder_pos_embed, weighted_decoder_pos_embed
