from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from models.two_stream_cwm import PathLike, VideoTensor

# from cwm.models import register_model
from models.two_stream_cwm.models.transformer.positional_encoding import sinusoidal_pos_embedding
from models.two_stream_cwm.models.transformer.utils import UsesTypicalInit, expand_pos_embed, trunc_normal_
from models.two_stream_cwm.models.two_stream_vmae import decoder, encoder
from models.two_stream_cwm.models.two_stream_vmae.params import TwoStreamEncoderParams
from models.two_stream_cwm.models.two_stream_vmae.two_stream.two_stream import TwoStreamTransformerParams
from models.two_stream_cwm.utils.io import load_from_json

NUM_CLASSES_8x8_RGB_POS = 2 + 3 * 8 * 8


def cast_lists_to_tuple(json_dict):
    enc_params = json_dict["encoder_params"]
    for level in ["primary", "secondary"]:
        json_dict["encoder_params"][level]["inp"]["inp_size"] = tuple(enc_params[level]["inp"]["inp_size"])
        json_dict["encoder_params"][level]["patch_size"] = tuple(enc_params[level]["patch_size"])

    return json_dict


@dataclass
class TwoStreamVMAEConfig:
    encoder_params: TwoStreamEncoderParams
    decoder_params: TwoStreamTransformerParams

    @classmethod
    def from_json(cls, path: PathLike) -> "TwoStreamVMAEConfig":
        return load_from_json(target_class=cls, path=path, caster=cast_lists_to_tuple)


EncoderType = encoder.TwoStreamVisionEncoder


class TwoStreamVMAE(ABC, nn.Module, UsesTypicalInit):
    """
    Two stream encoder + decoder
    """

    encoder: EncoderType

    def __init__(
        self,
        encoder_params: TwoStreamEncoderParams,
        decoder_params: TwoStreamTransformerParams,
    ) -> None:
        super().__init__()

        self.encoder = self._build_encoder(params=encoder_params)

        # two stream decoder
        self.decoder = decoder.TwoStreamDecoder(params=decoder_params)

        # Linear embedding to decoder dimensions
        self.encoder_to_decoder_primary = nn.Linear(
            in_features=encoder_params.primary.embed_dim,
            out_features=self.decoder.dim_primary,
            bias=False,
        )
        self.encoder_to_decoder_secondary = nn.Linear(
            in_features=encoder_params.secondary.embed_dim,
            out_features=self.decoder.dim_secondary,
            bias=False,
        )

        # mask tokens are learnable parameters that match decoder
        self.mask_token_primary = nn.Parameter(torch.zeros(1, 1, self.decoder.dim_primary))

        self.mask_token_secondary = nn.Parameter(torch.zeros(1, 1, self.decoder.dim_secondary))

        trunc_normal_(self.mask_token_primary)
        trunc_normal_(self.mask_token_secondary)

    @abstractmethod
    def _build_encoder(self, params: TwoStreamEncoderParams) -> EncoderType:
        raise NotImplementedError()

    @abstractmethod
    def _complete_decoder_inputs(
        self, x_primary: Tensor, x_secondary: Tensor, mask_primary: Tensor, mask_secondary: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError()

    def forward(
        self, x_primary: VideoTensor, mask_primary: Tensor, x_secondary: VideoTensor, mask_secondary: Tensor
    ) -> tuple[Tensor, Tensor]:
        # pass through encoder, which patchifies and gets pos embed
        # if bonus tokens provided, they are added here
        x_primary, x_secondary, mask_primary, mask_secondary = self.encoder(
            x_primary, mask_primary, x_secondary, mask_secondary
        )

        # linear embedding to decoder embed dims
        x_primary = self.encoder_to_decoder_primary(x_primary)
        x_secondary = self.encoder_to_decoder_secondary(x_secondary)

        # finally, pass through decoder
        num_visible_primary = x_primary.shape[1]
        num_visible_secondary = x_secondary.shape[1]

        (
            x_primary,
            x_secondary,
            x_primary_pe,
            x_secondary_pe,
        ) = self._complete_decoder_inputs(x_primary, x_secondary, mask_primary, mask_secondary)

        N_out_primary = x_primary.shape[1] - num_visible_primary
        N_out_secondary = x_secondary.shape[1] - num_visible_secondary

        y = self.decoder(
            x_primary,
            x_secondary,
            x_primary_pe,
            x_secondary_pe,
            return_last_n_tokens_primary=N_out_primary,
            return_last_n_tokens_secondary=N_out_secondary,
        )

        return y

    @classmethod
    def from_config_json(cls, config_path: Path):
        cfg = TwoStreamVMAEConfig.from_json(path=config_path)
        return cls(
            encoder_params=cfg.encoder_params,
            decoder_params=cfg.decoder_params,
        )


class TwoStreamMaskingVMAE(TwoStreamVMAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # positional embedding for decoder
        self.pos_embed_primary = sinusoidal_pos_embedding(
            positions=self.encoder.num_patches_primary,
            hidden_dim=self.decoder.dim_primary,
        )

        self.pos_embed_secondary = sinusoidal_pos_embedding(
            positions=self.encoder.num_patches_secondary,
            hidden_dim=self.decoder.dim_secondary,
        )

    def _build_encoder(self, params: TwoStreamEncoderParams) -> encoder.TwoStreamVisionEncoder:
        return encoder.TwoStreamVisionEncoder(params)

    def _complete_decoder_inputs(
        self, x_primary: Tensor, x_secondary: Tensor, mask_primary: Tensor, mask_secondary: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Completes decoder inputs by concatenating mask tokens after true tokens.

        Args:
            x_primary: primary stream tokens
            x_secondary: secondary stream tokens
            mask_primary: mask for primary stream
            mask_secondary: mask for secondary stream

        Returns: (x_primary, x_secondary, x_primary PE, x_secondary PE)
        """

        B, _, C = x_primary.shape
        _, _, D = x_secondary.shape
        pos_embed_primary = expand_pos_embed(self.pos_embed_primary, expand_as=x_primary)
        pos_embed_secondary = expand_pos_embed(self.pos_embed_secondary, expand_as=x_secondary)

        x_primary_pe_vis = pos_embed_primary[~mask_primary].reshape(B, -1, C)
        x_primary_pe_mask = pos_embed_primary[mask_primary].reshape(B, -1, C)
        x_secondary_pe_vis = pos_embed_secondary[~mask_secondary].reshape(B, -1, D)
        x_secondary_pe_mask = pos_embed_secondary[mask_secondary].reshape(B, -1, D)

        # get mask tokens
        mask_token_primary = self.mask_token_primary.expand(B, x_primary_pe_mask.size(1), -1)
        mask_token_secondary = self.mask_token_secondary.expand(B, x_secondary_pe_mask.size(1), -1)

        # create the full decoder inputs (positional embeddings separate)
        x_primary = torch.cat([x_primary, mask_token_primary], dim=1)
        x_secondary = torch.cat([x_secondary, mask_token_secondary], dim=1)
        x_primary_pe = torch.cat(
            [x_primary_pe_vis, x_primary_pe_mask],
            dim=1,
        )
        x_secondary_pe = torch.cat(
            [x_secondary_pe_vis, x_secondary_pe_mask],
            dim=1,
        )

        return x_primary, x_secondary, x_primary_pe, x_secondary_pe
