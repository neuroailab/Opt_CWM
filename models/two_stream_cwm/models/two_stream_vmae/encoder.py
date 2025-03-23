from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from models.two_stream_cwm.models.transformer.positional_encoding import sinusoidal_pos_embedding
from models.two_stream_cwm.models.transformer.utils import UsesTypicalInit, expand_pos_embed, trunc_normal_
from models.two_stream_cwm.models.transformer.vision_transformer import VideoPatchEmbed
from models.two_stream_cwm.models.two_stream_vmae.params import StreamParams, TwoStreamEncoderParams
from models.two_stream_cwm.models.two_stream_vmae.two_stream.two_stream import TwoStreamTransformer


class TwoStreamVisionEncoder(UsesTypicalInit, nn.Module):
    """A Vision transformer that has two communicating streams.
    Includes submodules for
    (1) tokenizing two input streams
    (2) running a TwoStreamTransformer "neck" that outputs from both streams.
    """

    def __init__(
        self,
        params: TwoStreamEncoderParams,
    ) -> None:
        super().__init__()

        self.params = params

        # tokenizer submodules and positional embeddings
        self._init_patch_embeddings()
        self._init_pos_embeddings()

        # transformer submodule
        self.two_stream_transformer = TwoStreamTransformer(
            params=self.params.transformer_params,
        )

        # norm layers
        self.primary_norm = params.transformer_params.norm_layer(params.primary.embed_dim)
        self.secondary_norm = params.transformer_params.norm_layer(params.secondary.embed_dim)

        self.apply(self._init_weights)

    def _init_patch_embeddings(self):
        self.patch_embed_primary = VideoPatchEmbed(
            params=self.params.primary.inp,
            patch_size=self.params.primary.patch_size,
            embed_dim=self.params.primary.embed_dim,
        )
        self.patch_embed_secondary = VideoPatchEmbed(
            params=self.params.secondary.inp,
            patch_size=self.params.secondary.patch_size,
            embed_dim=self.params.secondary.embed_dim,
        )

        self.num_patches_primary = self.patch_embed_primary.num_patches
        self.num_patches_secondary = self.patch_embed_secondary.num_patches

        self.num_frames_primary = self.params.primary.inp.num_frames
        self.num_frames_secondary = self.params.secondary.inp.num_frames

    def _init_pos_embeddings(self) -> None:
        primary_embed_shape = (
            1,
            self.num_patches_primary,
            self.params.primary.embed_dim,
        )
        secondary_embed_shape = (
            1,
            self.num_patches_secondary,
            self.params.secondary.embed_dim,
        )

        self.pos_embed_primary = (
            nn.Parameter(torch.zeros(primary_embed_shape))
            if self.params.use_learnable_pos_emb
            else sinusoidal_pos_embedding(positions=primary_embed_shape[1], hidden_dim=primary_embed_shape[2])
        )
        self.pos_embed_secondary = (
            nn.Parameter(torch.zeros(secondary_embed_shape))
            if self.params.use_learnable_pos_emb
            else sinusoidal_pos_embedding(positions=secondary_embed_shape[1], hidden_dim=secondary_embed_shape[2])
        )

        if self.params.use_learnable_pos_emb:
            trunc_normal_(self.pos_embed_primary)
            trunc_normal_(self.pos_embed_secondary)

    def get_mask_size(self, stream: Literal["primary", "secondary"] = "primary"):
        stream_params: StreamParams = getattr(self.params, stream)
        T, H, W = stream_params.inp.inp_size
        pt, ph, pw = stream_params.patch_size

        return T // pt, H // pw, W // pw

    @property
    def mask_size_primary(self):
        return self.get_mask_size("primary")

    @property
    def mask_size_secondary(self):
        return self.get_mask_size("secondary")

    @property
    def input_size_primary(self):
        return self.params.primary.inp.inp_size

    @property
    def input_size_secondary(self):
        return self.params.secondary.inp.inp_size

    @property
    def mask_size(self):
        return self.mask_size_primary

    @property
    def num_patches(self):
        return self.num_patches_primary

    def mask_tensor(self, z: Tensor, mask: Tensor, invert: bool = True) -> Tensor:
        B, N, C = z.shape
        assert mask.size(-1) == N, (mask.shape, N)

        mask_with = mask if not invert else ~mask
        return z[mask_with].reshape(B, -1, C)

    def mask_inputs_and_pos_embeddings(
        self,
        x_primary: Tensor,
        x_secondary: Tensor,
        x_primary_pe: Tensor,
        x_secondary_pe: Tensor,
        mask_primary: Tensor,
        mask_secondary: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x_primary = self.mask_tensor(x_primary, mask_primary)
        x_secondary = self.mask_tensor(x_secondary, mask_secondary)
        x_primary_pe = self.mask_tensor(x_primary_pe, mask_primary)
        x_secondary_pe = self.mask_tensor(x_secondary_pe, mask_secondary)
        return x_primary, x_secondary, x_primary_pe, x_secondary_pe

    def forward(
        self, x_primary: Tensor, mask_primary: Tensor, x_secondary: Tensor, mask_secondary: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Patchifies primary and secondary stream inputs, masks them,
        and passes through the TwoStreamTransformer.

        Supports passing only primary stream inputs. In this case,
        x_secondary will be a fully-masked zeros tensor.
        """
        # patchify the inputs
        x_primary = self.patch_embed_primary(x_primary)
        x_secondary = self.patch_embed_secondary(x_secondary)

        # get the positional embeddings, but don't add
        # the two_stream_transformer expects these as inputs
        x_primary_pe = expand_pos_embed(self.pos_embed_primary, expand_as=x_primary)
        x_secondary_pe = expand_pos_embed(self.pos_embed_secondary, expand_as=x_secondary)

        # apply masks to inputs and pos embeddings
        (
            x_primary,
            x_secondary,
            x_primary_pe,
            x_secondary_pe,
        ) = self.mask_inputs_and_pos_embeddings(
            x_primary=x_primary,
            x_secondary=x_secondary,
            x_primary_pe=x_primary_pe,
            x_secondary_pe=x_secondary_pe,
            mask_primary=mask_primary,
            mask_secondary=mask_secondary,
        )

        # apply two-stream transformer
        z_primary, z_secondary = self.two_stream_transformer(
            x_primary=x_primary,
            x_secondary=x_secondary,
            x_primary_pe=x_primary_pe,
            x_secondary_pe=x_secondary_pe,
        )

        z_primary = self.primary_norm(z_primary)
        z_secondary = self.secondary_norm(z_secondary)

        return z_primary, z_secondary, mask_primary, mask_secondary
