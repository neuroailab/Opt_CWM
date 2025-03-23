from typing import Optional

import numpy as np
import torch
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
from torch import optim as optim


def set_lr_and_wd(
    optimizer: torch.optim.Optimizer,
    lr_schedule: Optional[np.ndarray],
    wd_schedule: Optional[np.ndarray],
    global_it: int,
):
    if lr_schedule is None and wd_schedule is None:
        return

    for i, param_group in enumerate(optimizer.param_groups):
        if lr_schedule is not None:
            param_group["lr"] = lr_schedule[global_it] * param_group["lr_scale"]
        if wd_schedule is not None and param_group["weight_decay"] > 0:
            param_group["weight_decay"] = wd_schedule[global_it]


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, warmup_steps=-1
):
    """Lifted verbatim from
    https://github.com/rahulvenkk/cwm_release/blob/2e96800cf2e7762b3b3a19308fb0446d67e45779/cwm/utils.py#L358"""

    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    iter_per_len = iters / len(iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iter_per_len))
    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


if torch.cuda.device_count() > 0:
    use_tpu = False
else:
    import torch_xla.core.xla_model as xm

    print = xm.master_print
    use_tpu = True

try:
    from apex.optimizers import FusedAdam, FusedLAMB, FusedNovoGrad, FusedSGD

    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split(".")[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}
    all_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.0

            parameter_group_names[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": scale}
            parameter_group_vars[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": scale}

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

        all_names.append(name)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))

    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, "no_weight_decay"):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.0
    else:
        parameters = model.parameters()

    if "fused" in opt_lower:
        assert has_apex and torch.cuda.is_available(), "APEX and CUDA required for fused optimizers"

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, "opt_eps") and args.opt_eps is not None:
        opt_args["eps"] = args.opt_eps
    if hasattr(args, "opt_betas") and args.opt_betas is not None:
        opt_args["betas"] = args.opt_betas

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "sgd" or opt_lower == "nesterov":
        opt_args.pop("eps", None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == "momentum":
        opt_args.pop("eps", None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adamp":
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == "sgdp":
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "adafactor":
        if not args.lr:
            opt_args["lr"] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == "adahessian":
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == "rmsprop":
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == "rmsproptf":
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    # elif opt_lower == 'novograd':
    #     optimizer = NovoGrad(parameters, **opt_args)
    # elif opt_lower == 'nvnovograd':
    #     optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == "fusedsgd":
        opt_args.pop("eps", None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == "fusedmomentum":
        opt_args.pop("eps", None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == "fusedadam":
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == "fusedadamw":
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == "fusedlamb":
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == "fusednovograd":
        opt_args.setdefault("betas", (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == "lookahead":
            optimizer = Lookahead(optimizer)

    return optimizer
