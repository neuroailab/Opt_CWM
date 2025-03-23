from functools import partial

from torch import nn

from models.base_cwm import modeling_pretrain
from models.flow import flow_predictor, occ_predictor, perturber, soft_argmax
from models.opt_cwm import opt_cwm
from models.two_stream_cwm.models.flowcontrol import flow_fwd
from utils import dist_logging

logger = dist_logging.get_logger(__name__)


class _UpdateHookDict(dict):
    def __init__(self, *args, name=None, **kwargs):
        self.name = name
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(f"[{self.name}] Unrecognized field: {key}")

        if value != self[key]:
            logger.info(f"[{self.name}] Updated default field for {key}: {self[key]} -> {value}")

        return super().__setitem__(key, value)


def _get_base_cwm(base_cwm_args):
    """
    Scaffold base_cwm with train-time configurations.

    Arguments:
      base_cwm_args: Additional kwargs to change from default settings.
      highres: Flag to enable positional embedding interpolation,
        allowing for higher resolution inputs.

    Returns:
      (nn.Module): The base_cwm model.
    """
    cls = modeling_pretrain.PretrainVisionTransformer

    default_args = dict(
        img_size=256,
        patch_size=(8, 8),
        num_frames=2,
        tubelet_size=1,
        use_learnable_pos_emb=True,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        encoder_in_chans=3,
        decoder_num_classes=None,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=0.0,
        decoder_embed_dim=768,
        decoder_num_heads=12,
        decoder_depth=12,
        mlp_ratio=4,
        qkv_bias=True,
        k_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    default_args = _UpdateHookDict(**default_args, name=cls.__name__)
    for k, v in base_cwm_args.items():
        default_args[k] = v

    model = cls(**default_args)

    return model


def _get_soft_argmax(softargmax_args):
    """
    Scaffold soft argmax with train-time configurations.

    Arguments:
      softargmax_args: Additional kwargs to change from default settings.

    Returns:
      (nn.Module): The soft argmax model.
    """
    cls = soft_argmax.SoftArgmax
    default_args = dict(reduce_fn="l1", inv_temp=200)
    default_args = _UpdateHookDict(**default_args, name=cls.__name__)

    for k, v in softargmax_args.items():
        default_args[k] = v

    return cls(**default_args)


def _get_gauss_perturber(perturber_args, input_dim=512):
    """
    Scaffold gaussian perturber with train-time configurations.

    Arguments:
      perturber_args: Additional kwargs to change from default settings.
      input_dim: MLP input dimension. Note that this is determined
        by the base_cwm encoder embedding dimension, and
        therefore not separately configurable.
      highres: Flag to enable perturbation scaling. This is
        enabled when using highres base_cwm.

    Returns:
      (nn.Module): The gaussian perturber model.
    """
    cls = perturber.GaussPerturber
    default_args = dict(pert_size=20, hidden_dim=512, input_dim=input_dim)
    default_args = _UpdateHookDict(**default_args, name=cls.__name__)

    for k, v in perturber_args.items():
        default_args[k] = v

    return cls(**default_args)


def _get_occ_predictor(occ_args):
    """
    Scaffold occlusion predictor with train-time configurations.

    Arguments:
      occ_args: Additional kwargs to change from default settings.

    Returns:
      (nn.Module): The occlusion predictor model.
    """
    cls = occ_predictor.OccPredictor
    default_args = dict(spatial_reduction="max", masking_reduction="mean", thresh=0.05)

    for k, v in occ_args.items():
        default_args[k] = v

    return cls(**default_args)


def get_flow_predictor(model_args):
    """
    Scaffold flow predictor and submodules
    with train-time configurations.

    Arguments:
      model_args: Additional kwargs to change from default settings.

    Returns:
      (nn.Module): The flow predictor model.
    """
    base_cwm_module = _get_base_cwm(model_args.get("base_cwm", {}))
    softargmax_module = _get_soft_argmax(model_args.get("soft_argmax", {}))

    pert_in_dim = base_cwm_module.encoder.embed_dim

    perturber_module = _get_gauss_perturber(model_args.get("gauss_perturber", {}), input_dim=pert_in_dim)
    occ_module = _get_occ_predictor(model_args.get("occ_predictor", {}))

    cls = flow_predictor.FlowPredictor

    default_args = dict(masking_iters=1, masking_ratio=0.9, zoom_iters=0)
    default_args = _UpdateHookDict(**default_args, name=cls.__name__)

    for k, v in model_args.get("flow_predictor", {}).items():
        default_args[k] = v

    return cls(
        cwm_model=base_cwm_module,
        perturber=perturber_module,
        occ_module=occ_module,
        softargmax_module=softargmax_module,
        **default_args,
    )


def _get_two_stream_cwm(model_args, device_id=None):
    return flow_fwd.FlowFwd.from_config(model_args.build.two_stream_cwm_config_path, device=device_id)


def get_opt_cwm(model_args, device_id=None):
    """
    Scaffold opt_cwm and submodules
    with train-time configurations.

    Arguments:
      model_args: Additional kwargs to change from default settings.
      device_id: Current device.

    Returns:
      (nn.Module): The opt_cwm model.
    """
    flow = get_flow_predictor(model_args)
    two_stream_cwm = _get_two_stream_cwm(model_args, device_id=device_id)

    cls = opt_cwm.OptCWM

    default_args = dict(n_flow_pts=8)
    default_args = _UpdateHookDict(**default_args, name=cls.__name__)

    for k, v in model_args.get("opt_cwm", {}).items():
        default_args[k] = v

    return cls(flow_predictor=flow, recon_cwm_model=two_stream_cwm, **default_args)
