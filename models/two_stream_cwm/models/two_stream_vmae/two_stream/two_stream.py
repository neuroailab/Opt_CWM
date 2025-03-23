from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from models.two_stream_cwm.models.transformer.utils import UsesTypicalInit
from models.two_stream_cwm.models.two_stream_vmae.two_stream.blocks import (
    ConjoinedTwoStreamBlock,
    TwoStreamBlock,
    TwoStreamBlockParams,
)
from models.two_stream_cwm.models.two_stream_vmae.types import ConnectivityPattern


### Specify block type by string ###
def _get_connectivity_params(
    self_primary=True,
    self_secondary=False,
    cross_onto_primary=True,
    cross_onto_secondary=True,
) -> dict:
    return {
        "skip_self_attention_primary": not self_primary,
        "skip_self_attention_secondary": not self_secondary,
        "skip_secondary_onto_primary": not cross_onto_primary,
        "skip_primary_onto_secondary": not cross_onto_secondary,
    }


block_connectivity_legend = {
    k: _get_connectivity_params(*v)
    for k, v in {
        "_-_": (False, False, False, False),
        "o-_": (True, False, False, False),
        "_-o": (False, True, False, False),
        "o-o": (True, True, False, False),
        "_<-_": (False, False, True, False),
        "_->_": (False, False, False, True),
        "_<->_": (False, False, True, True),
        "o<-_": (True, False, True, False),
        "o->_": (True, False, False, True),
        "o<->_": (True, False, True, True),
        "_<-o": (False, True, True, False),
        "_->o": (False, True, False, True),
        "_<->o": (False, True, True, True),
        "o<-o": (True, True, True, False),
        "o->o": (True, True, False, True),
        "o<->o": (True, True, True, True),
    }.items()
}


@dataclass
class StreamOutputParams:
    should_output: bool = False
    num_classes: int = 0


@dataclass
class TwoStreamTransformerParams:
    block_params: TwoStreamBlockParams
    primary_output: StreamOutputParams = field(default_factory=StreamOutputParams)
    secondary_output: StreamOutputParams = field(default_factory=StreamOutputParams)

    block_func: type[nn.Module] = TwoStreamBlock
    conn_patterns: ConnectivityPattern | list[ConnectivityPattern] = "o<->o"
    drop_path_rate: float = 0.0
    depth: int = 4
    norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6)
    apply_pos_emb_all_layers: bool = False


class TwoStreamTransformer(UsesTypicalInit, nn.Module):
    def __init__(
        self,
        params: TwoStreamTransformerParams,
    ) -> None:
        """
        Transformer "neck" that takes in two sets of tokens (and optional positional
        encodings for each) and processes them with pairs of blocks that may be
        conjoined.

        Can output from either stream or both.
        """
        super().__init__()

        # bail early if there are no layers to create
        if params.depth < 1:
            raise ValueError("Cannot create a Transformer with fewer than 1 blocks")

        self.params = params
        self.norm_layer = self.params.norm_layer

        # stochastic depth decay rule
        drop_path_rates = [x.item() for x in torch.linspace(0, params.drop_path_rate, params.depth)]

        # determine the connectivity for each block
        self._parse_conn_patterns()

        # create the blocks
        self.blocks = nn.ModuleList(
            [
                ConjoinedTwoStreamBlock(
                    params.block_params,
                    drop_path=drop_path_rates[idx],
                    **block_connectivity_legend[self.conn_patterns[idx]],
                )
                for idx in range(params.depth)
            ]
        )

        # primary and secondary stream dims
        self.dim_primary = params.block_params.primary.dim
        self.dim_secondary = params.block_params.secondary.dim

        # create the output
        self.output_layers = self._build_output_config_and_layers(
            output_primary_stream=self.params.primary_output.should_output,
            output_secondary_stream=self.params.secondary_output.should_output,
            num_classes_primary=self.params.primary_output.num_classes,
            num_classes_secondary=self.params.secondary_output.num_classes,
        )

        # how to handle positional embedding
        self.apply_pos_emb_all_layers = params.apply_pos_emb_all_layers

        # apply init weights method from UsesTypicalInit
        self.apply(self._init_weights)

    def _parse_conn_patterns(self):
        """Sanitize and parse connectivity patterns.

        Args:
            conn_patterns: Either a single connectivity pattern to use for all
                layers, or a list with one entry per layer.
        """
        if isinstance(self.params.conn_patterns, str):
            assert self.params.conn_patterns in block_connectivity_legend.keys()
            self.conn_patterns = [self.params.conn_patterns] * self.params.depth
        else:
            assert len(self.params.conn_patterns) == self.params.depth, "Must pass one connectivity pattern per layer"
            assert all((c in block_connectivity_legend.keys() for c in self.params.conn_patterns))
            self.conn_patterns = self.params.conn_patterns

    def _build_output_config_and_layers(
        self,
        num_classes_primary: int = 1,
        num_classes_secondary: int = 0,
        output_primary_stream: bool = True,
        output_secondary_stream: bool = False,
    ) -> nn.ModuleDict:
        self._output_primary_stream = output_primary_stream
        self._output_secondary_stream = output_secondary_stream

        output_layers = nn.ModuleDict(
            {
                "norm_primary": self.norm_layer(self.dim_primary) if output_primary_stream else nn.Identity(),
                "norm_secondary": self.norm_layer(self.dim_secondary) if output_secondary_stream else nn.Identity(),
                "head_primary": (
                    nn.Linear(self.dim_primary, num_classes_primary) if num_classes_primary > 0 else nn.Identity()
                ),
                "head_secondary": (
                    nn.Linear(self.dim_secondary, num_classes_secondary) if num_classes_secondary > 0 else nn.Identity()
                ),
            }
        )

        return output_layers

    def _forward_features(self, queries, keys, queries_pe, keys_pe) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            queries, keys = block(queries, keys, queries_pe, keys_pe)

        return queries, keys

    # output layers
    def apply_output_layers(
        self,
        x_primary,
        x_secondary,
        n_primary: int | None = None,
        n_secondary: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        if n_primary is not None:
            x_primary = x_primary[:, -n_primary:]
        if n_secondary is not None:
            x_secondary = x_secondary[:, -n_secondary:]

        if self._output_primary_stream:
            x_primary = self.output_layers["head_primary"](self.output_layers["norm_primary"](x_primary))

        if self._output_secondary_stream:
            x_secondary = self.output_layers["head_secondary"](self.output_layers["norm_secondary"](x_secondary))

        return x_primary, x_secondary

    def forward(
        self,
        x_primary: Tensor,
        x_secondary: Tensor,
        x_primary_pe: Tensor | None = None,
        x_secondary_pe: Tensor | None = None,
        return_last_n_tokens_primary: int | None = None,
        return_last_n_tokens_secondary: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Runs the two streams of inputs (and their positional embeddings)
        through all the conjoined two stream blocks
        """
        # Apply positional embedding only once, then set the pe variables to None so
        # they don't get added at any future steps.
        if not self.apply_pos_emb_all_layers:
            if x_primary_pe is not None:
                x_primary = x_primary + x_primary_pe
            if x_secondary_pe is not None:
                x_secondary = x_secondary + x_secondary_pe
            x_primary_pe, x_secondary_pe = None, None

        # layers
        x_primary, x_secondary = self._forward_features(
            queries=x_primary,
            keys=x_secondary,
            queries_pe=x_primary_pe,
            keys_pe=x_secondary_pe,
        )

        # output norm and head
        x_primary, x_secondary = self.apply_output_layers(
            x_primary,
            x_secondary,
            n_primary=return_last_n_tokens_primary,
            n_secondary=return_last_n_tokens_secondary,
        )

        return x_primary, x_secondary
