from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor


@dataclass
class MLPParams:
    in_features: int
    hidden_features: int | None = None
    out_features: int | None = None
    act_layer: type[nn.Module] = nn.GELU
    bias: bool = False
    drop_rate: float = 0.0


class MLP(nn.Module):
    """Very similar to timm.layers.mlp.Mlp, except that we skip the first dropout"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        bias: bool = True,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        # if out_features is not provided, assume it should be the same as in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features, bias=bias)
        self.drop = nn.Dropout(p=drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        """Implements:
        dropout(
            fc2(
                activation(
                    fc1(inp)
                )
            )
        )
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
