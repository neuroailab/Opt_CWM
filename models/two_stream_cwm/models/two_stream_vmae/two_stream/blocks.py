from abc import abstractmethod
from dataclasses import asdict, dataclass

import torch.nn as nn
from torch import Tensor

from models.two_stream_cwm.models.layers.drop import DropPath
from models.two_stream_cwm.models.layers.mlp import MLP
from models.two_stream_cwm.models.transformer.transformer import Attention, BlockParams
from models.two_stream_cwm.models.two_stream_vmae.two_stream.cross_attention import OneWayCrossAttention


@dataclass
class TwoStreamBlockParams:
    """
    Parameters for a block with two streams.

    The streams need not have the same embedding dimension,
    but do by default.
    """

    primary: BlockParams
    secondary: BlockParams


class TwoStreamBlock(nn.Module):
    def __init__(
        self,
        params: TwoStreamBlockParams,
        drop_path: float = 0.0,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        Abstract base class for a pair of transformer blocks,
        corresponding to two token streams.

        Subclasses allow the two streams to pass (uni- or bi-directional) messages
        via OneWayCrossAttention, or not communicate at all.

        Additionally, one stream can have no block at this layer (passthrough.)
        """
        super().__init__()
        self.params = params
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # whether to skip applying positional embedding at first layer
        self.skip_first_layer_pe = skip_first_layer_pe

    def apply_positional_embedding(self, *args, **kwargs) -> Tensor:
        """
        Abstract method to allow other ways of applying positional embedding besides
        adding it.
        """
        return self._add_positional_embedding(*args, **kwargs)

    def _add_positional_embedding(self, x: Tensor, pos_embed: Tensor | None = None) -> Tensor:
        """
        If a positional embedding was passed, add it to input.
        Otherwise, do nothing.
        """
        if pos_embed is not None:
            assert pos_embed.shape[-2:] == x.shape[-2:], (
                "Positional embedding and input tensor must have same number of tokens " "and dim"
            )
            return x + pos_embed
        return x

    @abstractmethod
    def apply_self_attention_layer(
        self,
        queries: Tensor,
        keys: Tensor,
        queries_pe: Tensor | None = None,
        keys_pe: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Primary and/or secondary streams may have an initial self attention layer
        """
        raise NotImplementedError

    @abstractmethod
    def apply_cross_attention_layer(
        self,
        queries: Tensor,
        keys: Tensor,
        queries_pe: Tensor | None = None,
        keys_pe: Tensor | None = None,
        output_primary_stream: bool = True,
    ) -> Tensor:
        """
        Cross attention in which keys attend to queries
        """
        raise NotImplementedError

    @abstractmethod
    def apply_mlp_layer(self, queries: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        queries_pe: Tensor | None = None,
        keys_pe: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Takes inputs from two streams and processes each.

        queries are the inputs from the "primary" stream.
        keys are the inputs from the "secondary" stream.

        Depending on the module config, the two streams may interact
        or simply pass-through.
        """
        # check if either of these is empty; if so, pass through
        if queries.shape[1] == 0 or keys.shape[1] == 0:
            return queries, keys

        # (1) initial self-attention layer on queries and/or keys
        queries, keys = self.apply_self_attention_layer(queries, keys, queries_pe, keys_pe)

        # (2) cross attention of secondary onto primary stream
        queries = self.apply_cross_attention_layer(queries, keys, queries_pe, keys_pe, output_primary_stream=True)

        # (3) mlp on each stream
        queries, keys = self.apply_mlp_layer(queries, keys)

        # (4) cross attention of primary onto secondary stream
        keys = self.apply_cross_attention_layer(keys, queries, keys_pe, queries_pe, output_primary_stream=False)

        return queries, keys


class ParallelStreamBlock(TwoStreamBlock):
    def __init__(
        self,
        params: TwoStreamBlockParams,
        skip_self_attention_primary: bool = False,
        skip_self_attention_secondary: bool = False,
        **kwargs,
    ) -> None:
        """A two-stream block that has no cross attention between the two streams."""
        super().__init__(params, **kwargs)

        # self attn and mlp on streams
        self.self_attn_layers_primary = self._build_self_attention_layers(
            params.primary, passthrough=skip_self_attention_primary
        )
        self.self_attn_layers_secondary = self._build_self_attention_layers(
            params.secondary, passthrough=skip_self_attention_secondary
        )
        self._skip_self_attention_primary = skip_self_attention_primary
        self._skip_self_attention_secondary = skip_self_attention_secondary

    def _build_self_attention_layers(
        self,
        params: BlockParams,
        passthrough: bool = False,
    ) -> nn.ModuleDict:
        if passthrough:
            layer_keys = [
                "self_attn",
                "mlp",
                "norm_before_self_attn",
                "norm_before_mlp",
            ]
            return nn.ModuleDict({k: nn.Identity() for k in layer_keys})

        return nn.ModuleDict(
            {
                "self_attn": Attention(params.dim, proj_drop=params.drop_rate, **asdict(params.attn)),
                "mlp": MLP(
                    in_features=params.dim,
                    hidden_features=params.hidden_dim,
                    act_layer=params.act_layer,
                    drop_rate=params.drop_rate,
                ),
                "norm_before_self_attn": params.norm_layer(params.dim),
                "norm_before_mlp": params.norm_layer(params.dim),
            }
        )

    def apply_self_attention_layer(
        self,
        queries: Tensor,
        keys: Tensor,
        queries_pe: Tensor | None = None,
        keys_pe: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Primary and/or secondary streams may have an initial self attention layer
        """
        if self.skip_first_layer_pe:
            queries = self.self_attn_layers_primary["norm_before_self_attn"](queries)
            keys = self.self_attn_layers_secondary["norm_before_self_attn"](keys)

            queries = self.self_attn_layers_primary["self_attn"](queries)
            keys = self.self_attn_layers_secondary["self_attn"](keys)
        else:  # make sure it's a true passthrough if skipping one stream
            queries = self.apply_positional_embedding(
                queries, queries_pe if not self._skip_self_attention_primary else None
            )
            keys = self.apply_positional_embedding(keys, keys_pe if not self._skip_self_attention_secondary else None)

            q = self.self_attn_layers_primary["norm_before_self_attn"](queries)
            k = self.self_attn_layers_secondary["norm_before_self_attn"](keys)

            attn_primary = self.self_attn_layers_primary["self_attn"](q)
            attn_secondary = self.self_attn_layers_secondary["self_attn"](k)

            if not self._skip_self_attention_primary:
                queries = attn_primary + queries
            else:
                queries = attn_primary

            if not self._skip_self_attention_secondary:
                keys = attn_secondary + keys
            else:
                keys = attn_secondary

        return queries, keys

    def apply_cross_attention_layer(
        self,
        queries: Tensor,
        keys: Tensor,
        queries_pe: Tensor | None = None,
        keys_pe: Tensor | None = None,
        output_primary_stream: bool = True,
    ) -> Tensor:
        """
        Cross attention in which keys attend to queries
        """
        return queries

    def apply_mlp_layer(self, queries: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
        # make sure it's a true passthrough if skipping a stream
        mlp_out_primary = self.self_attn_layers_primary["mlp"](
            self.self_attn_layers_primary["norm_before_mlp"](queries)
        )
        mlp_out_secondary = self.self_attn_layers_secondary["mlp"](
            self.self_attn_layers_secondary["norm_before_mlp"](keys)
        )

        if not self._skip_self_attention_primary:
            queries = mlp_out_primary + queries
        else:
            queries = mlp_out_primary

        if not self._skip_self_attention_secondary:
            keys = mlp_out_secondary + keys
        else:
            keys = mlp_out_secondary

        return queries, keys


class ConjoinedTwoStreamBlock(ParallelStreamBlock):
    def __init__(
        self,
        params: TwoStreamBlockParams,
        skip_secondary_onto_primary: bool = False,
        skip_primary_onto_secondary: bool = False,
        **kwargs,
    ) -> None:
        """A two-stream block that has cross attention between the two streams"""

        # initialize self-attention and mlps
        super().__init__(params, **kwargs)

        self.cross_attn_secondary_onto_primary_layers = self._build_cross_attention_layers(
            params,
            output_primary_stream=True,
            passthrough=skip_secondary_onto_primary,
        )
        self.cross_attn_primary_onto_secondary_layers = self._build_cross_attention_layers(
            params,
            output_primary_stream=False,
            passthrough=skip_primary_onto_secondary,
        )
        self._skip_secondary_onto_primary = skip_secondary_onto_primary
        self._skip_primary_onto_secondary = skip_primary_onto_secondary

    def _build_cross_attention_layers(
        self,
        params: TwoStreamBlockParams,
        output_primary_stream: bool,
        passthrough: bool = False,
    ):
        if passthrough:
            layer_keys = [
                "embed_primary",
                "embed_secondary",
                "embed_values",
                "cross_attn",
                "norm_before_cross_attn",
            ]
            return nn.ModuleDict({k: nn.Identity() for k in layer_keys})

        dim_primary, dim_secondary = params.primary.dim, params.secondary.dim
        if dim_primary != dim_secondary:
            do_linear_embedding = True
            if output_primary_stream:
                dim = dim_primary
            else:
                dim = dim_secondary
        else:
            do_linear_embedding = False
            dim = dim_primary

        _attn_params = params.primary.attn if output_primary_stream else params.secondary.attn

        layers = nn.ModuleDict(
            {
                "embed_primary": nn.Linear(dim_primary, dim, bias=False) if do_linear_embedding else nn.Identity(),
                "embed_secondary": nn.Linear(dim_secondary, dim, bias=False) if do_linear_embedding else nn.Identity(),
                "embed_values": (
                    nn.Linear(
                        dim_secondary if output_primary_stream else dim_primary,
                        dim,
                        bias=False,
                    )
                    if do_linear_embedding
                    else nn.Identity()
                ),
                # cross attn
                "cross_attn": OneWayCrossAttention(
                    dim,
                    proj_drop=params.primary.drop_rate,
                    downsample_factor=params.primary.downsample_factor,
                    **asdict(_attn_params),
                ),
                "norm_before_cross_attn_q": params.primary.norm_layer(dim),
                "norm_before_cross_attn_k": params.primary.norm_layer(dim),
                "norm_before_cross_attn_v": params.primary.norm_layer(dim),
            }
        )

        return layers

    def apply_cross_attention_layer(
        self,
        queries: Tensor,
        keys: Tensor,
        queries_pe: Tensor | None = None,
        keys_pe: Tensor | None = None,
        output_primary_stream: bool = True,
    ) -> Tensor:
        """
        Cross attention in which keys attend to queries.

        This function always outputs the modified queries, so if it is meant to
        apply to the secondary stream, the first two args should be swapped in the
        function call
        """
        # check whether this is a passthrough
        if output_primary_stream and self._skip_secondary_onto_primary:
            return queries
        elif (not output_primary_stream) and self._skip_primary_onto_secondary:
            return queries

        # apply pos embedding
        q = self.apply_positional_embedding(queries, queries_pe)
        k = self.apply_positional_embedding(keys, keys_pe)

        # linear embedding to common dimension
        layers = (
            self.cross_attn_secondary_onto_primary_layers
            if output_primary_stream
            else self.cross_attn_primary_onto_secondary_layers
        )
        q = layers["embed_primary" if output_primary_stream else "embed_secondary"](q)
        k = layers["embed_secondary" if output_primary_stream else "embed_primary"](k)
        v = layers["embed_values"](keys)

        # norm inputs + cross attention
        attn_out = layers["cross_attn"](
            q=layers["norm_before_cross_attn_q"](q),
            k=layers["norm_before_cross_attn_k"](k),
            v=layers["norm_before_cross_attn_v"](v),
        )
        queries = queries + attn_out

        return queries
