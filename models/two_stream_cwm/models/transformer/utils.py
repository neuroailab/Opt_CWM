import os
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from models.two_stream_cwm import PathLike

# constants and/or hyperparams
TRUNC_NORMAL_SIGMA = 0.02


class UsesTypicalInit:
    """A 'typical' init means that bias terms are initialized to 0, linear layers get
    Xavier init, and layernorm bias and weight get set to 0 and 1, respectively

    Classes can inherit from this mixin to incorporate this weight init method
    """

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def trunc_normal_(tensor: Tensor, mean: float = 0.0, std: float = TRUNC_NORMAL_SIGMA) -> None:
    """Intializes a tensor with values drawn from a truncated normal distribution. We
    specify that the cutoffs for truncation should be +/- 1 * std (the default is
    effectively +/- 2 * std).

    As far as I can tell this is a bug, introduced by BEiT incorrectly copying the
    BERT code.

    Args:
        tensor: input tensor
        mean: mean of normal distribution. Defaults to 0.0.
        std: std of normal distribution. Defaults to 0.02.
    """
    return nn.init._no_grad_trunc_normal_(tensor=tensor, mean=mean, std=std, a=-std, b=std)


def is_flash_attention_available() -> bool:
    """Checks if flash attention is available in the current environment."""
    has_fused_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    # check if
    try:
        use_fused_attn = int(os.environ.get("CWM_USE_FUSED_ATTN", "1"))
    except TypeError as e:
        print("CWM_USE_FUSED_ATTN set to invalid value. Use only '0' or '1'")
        print(e)
        use_fused_attn = 0

    return bool(has_fused_attn and use_fused_attn)


def load_legacy_checkpoint(
    model: nn.Module, ckpt_path: PathLike, map_location: str = "cpu"
) -> nn.modules.module._IncompatibleKeys:
    """Perform a specific kind of surgery on checkpoint to fuse attention
    bias terms back into a concatenated matrix.

    For example, biases from these checkpoints will have a separate q_bias term
    and v_bias term. Newer models just concatenate the q, k, and v bias terms
    into a single long vector. This function re-concatenates and renames the keys
    to allow model.load_state_dict() to succeed

    Args:
        model: model to load ckpt state dict into
        ckpt_path: str or Path to the checkpoint file
    """
    ckpt_path = Path(ckpt_path)
    model_ckpt = torch.load(ckpt_path, map_location=map_location)["model"]

    def _is_attn_bias(key: str, prefix: str) -> bool:
        return key.startswith(prefix) and "attn" in key and "bias" in key and "proj" not in key

    for prefix in ["encoder.blocks.", "decoder.blocks."]:
        keys = [k for k in model_ckpt.keys() if _is_attn_bias(k, prefix)]

        # find highest block number
        block_numbers = [int(k.split(prefix)[-1].split(".")[0]) for k in keys]
        highest_block_number = max(block_numbers)

        for block in range(highest_block_number + 1):
            q_bias_key = f"{prefix}{block}.attn.q_bias"
            v_bias_key = f"{prefix}{block}.attn.v_bias"

            q_bias = model_ckpt.pop(q_bias_key)
            v_bias = model_ckpt.pop(v_bias_key)
            k_bias = torch.zeros_like(v_bias, dtype=v_bias.dtype)

            # concatenate into single tensor
            qkv_bias = torch.cat((q_bias, k_bias, v_bias))

            # add to dictionary
            model_ckpt[f"{prefix}{block}.attn.qkv.bias"] = qkv_bias

    status = model.load_state_dict(model_ckpt)
    return status


def expand_pos_embed(pos_embed: Tensor, expand_as: Tensor):
    """_summary_

    Args:
        pos_embed: _description_
        expand_as: _description_

    Returns:
        _description_

    NB:
        It's critical to detach the positional embedding so that you don't get the
        dreaded 'backprop through graph twice error'. TODO: more detail once we
        understand why it happens
    """
    batch_size = expand_as.shape[0]
    return pos_embed.expand(batch_size, -1, -1).type_as(expand_as).to(expand_as.device).clone().detach()
