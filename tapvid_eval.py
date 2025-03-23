import datetime
import os
import sys

import torch
import torch.distributed as dist
from tqdm import tqdm

from data.tapvid import dataset as tapvid_dataset
from external.sea_raft import raft_wrapper
from models import builder
from utils import dist_logging, options, tapvid_eval_utils, utils

logger = dist_logging.get_logger(__name__)


def _setup(meta_args, data_args, model_args):
    torch.backends.cuda.enable_flash_sdp(True)

    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=meta_args.nccl_timeout))

    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)

    utils.set_seed(meta_args.seed)

    if model_args.build.get("raft_config_path"):
        logger.info("Running TAP-Vid Evaluation with Distilled SEA-RAFT.")
        logger.info(f"Loading SEA-RAFT configurations from {model_args.build.raft_config_path}")

        model = raft_wrapper.RAFTWrapper(model_args.build.raft_config_path)

        if model_args.build.raft_ckpt_path is not None:
            logger.info("RAFT Checkpoint path provided. Loading directly from path.")
            assert os.path.exists(model_args.build.raft_ckpt_path)
            model.raft.load_state_dict(torch.load(model_args.build.raft_ckpt_path, map_location="cpu"))
        else:
            model.load_pretrained(force=model_args.build.force)
    else:
        logger.info("Running TAP-Vid Evaluation with Opt-CWM FlowPredictor.")

        model = builder.get_flow_predictor(model_args)
        model.load_pretrained(highres=model_args.build.highres, force=model_args.build.force)

    model = model.cuda()

    loader = tapvid_dataset.get_tapvid_loader(
        dataset=data_args.tapvid.dataset,
        resolution=data_args.tapvid.resolution,
        frame_delta=data_args.tapvid.frame_delta,
        path_to_pkl=data_args.tapvid.path_to_pkl,
        batch_size=data_args.batch_size,
        debug=data_args.tapvid.debug,
        num_workers=data_args.num_workers,
    )

    return model, loader


def _eval_batch(model, batch):
    num_groups = batch["num_video_groups"]

    all_pred_occlusions = []
    all_expec_pred_locs = []
    all_argmax_pred_locs = []
    all_multi_scale_pred_locs = []
    all_uids = []

    for g in range(num_groups):
        videos, points, video_name, occluded, uid = (
            batch[f"{g}_videos"],
            batch[f"{g}_points"],
            batch[f"{g}_video_name"],
            batch[f"{g}_occluded"],
            batch[f"{g}_uid"],
        )

        videos = videos.cuda()
        points = points.long().cuda()

        pt0, _ = torch.unbind(points, 1)

        with torch.cuda.amp.autocast(enabled=True):
            model_out = model(videos, pt0)

        all_pred_occlusions.append(model_out["pred_occ"])
        all_expec_pred_locs.append(model_out["expec_pred_pixel_loc"])
        all_argmax_pred_locs.append(model_out["argmax_pred_pixel_loc"])
        all_multi_scale_pred_locs.append(model_out["multi_scale_pred_pixel_loc"])
        all_uids.append(uid)

    pred_occlusion = torch.cat(all_pred_occlusions, 0)
    expec_pred_loc = torch.cat(all_expec_pred_locs, 0)
    argmax_pred_loc = torch.cat(all_argmax_pred_locs, 0)
    multi_scale_pred_locs = torch.cat(all_multi_scale_pred_locs, 0)
    uids = torch.cat(all_uids, 0).cuda()

    return pred_occlusion, expec_pred_loc, argmax_pred_loc, multi_scale_pred_locs, uids


@torch.no_grad()
def eval(model, loader, cache_dir=None, evaluator_debug=False, sync_results_every=15):
    """Runs TAP-Vid Evaluation.

    Arguments:
      model: FlowPredictor model, or SEA-RAFT model if running with distilled.
      loader: TAP-Vid Dataloader.
      cache_dir: Directory to save evaluation metrics and cache.
      evaluator_debug: If set, runs async evaluator on main thread.
      sync_results_every: Period for collecting results across devices and submitting job.
    """

    model.eval()

    if dist.get_rank() == 0:
        evaluator = tapvid_eval_utils.AsyncTAPVidEvaluator(
            dataset=loader.dataset, debug=evaluator_debug, cache_dir=cache_dir
        )

    collect_expec, collect_argmax, collect_multi_scale, collect_uids = [], [], [], []
    collect_pred_occ = []

    for i, batch in tqdm(
        enumerate(loader), total=len(loader), desc=f"(rank 0) DAVIS Evaluation", disable=int(os.environ["LOCAL_RANK"])
    ):

        pred_occ, expec, argmax, multi_scale, uid = _eval_batch(model, batch)

        collect_pred_occ.append(pred_occ)
        collect_expec.append(expec)
        collect_argmax.append(argmax)
        collect_multi_scale.append(multi_scale)
        collect_uids.append(uid)

        if i % sync_results_every == 0 or i == len(loader) - 1:

            pred_occ = torch.cat(collect_pred_occ, 0)
            expecs = torch.cat(collect_expec, 0).flip(-1)  # (x, y)
            argmaxs = torch.cat(collect_argmax, 0).flip(-1)  # (x, y)
            multi_scales = torch.cat(collect_multi_scale, 0).flip(-1)  # (x, y)

            uids = torch.cat(collect_uids, 0)

            assert (
                pred_occ.size(0) == expecs.size(0) == argmaxs.size(0) == multi_scales.size(0) == uids.size(0)
            ), f"{pred_occ.size(0)}, {expecs.size(0)}, {argmaxs.size(0)}, {multi_scales.size(0)}, {uids.size(0)} are not equal!"

            pred_occ = utils.gather_tensor(pred_occ).cpu().numpy()
            expecs = utils.gather_tensor(expecs).cpu().numpy()
            argmaxs = utils.gather_tensor(argmaxs).cpu().numpy()
            multi_scales = utils.gather_tensor(multi_scales).cpu().numpy()
            uids = utils.gather_tensor(uids).cpu().numpy()

            # clear for next sync
            collect_pred_occ.clear()
            collect_expec.clear()
            collect_argmax.clear()
            collect_multi_scale.clear()
            collect_uids.clear()

            if dist.get_rank() == 0:
                evaluator.submit_eval_job(
                    {
                        "occlusion": pred_occ,
                        "expec": expecs,
                        "argmax": argmaxs,
                        "multi_scale": multi_scales,
                        "uids": uids,
                    }
                )

    dist.barrier()

    metrics = {}
    if dist.get_rank() == 0:
        metrics = evaluator.final_result()
        if cache_dir is not None:
            torch.save(metrics, os.path.join(cache_dir, f"tapvid_val_metrics.pt"))

    return metrics


if __name__ == "__main__":
    opt_cmd = options.parse_arguments(sys.argv[1:])
    eval_cfg = options.set(opt_cmd=opt_cmd, verbose=False)

    data_args = eval_cfg.data_args
    meta_args = eval_cfg.meta_args
    model_args = eval_cfg.model_args

    model, loader = _setup(meta_args, data_args, model_args)

    if meta_args.cache_dir is not None and dist.get_rank() == 0:
        os.makedirs(meta_args.cache_dir, exist_ok=True)

    eval(
        model,
        loader,
        meta_args.cache_dir,
        meta_args.tapvid_evaluator.debug,
        meta_args.tapvid_evaluator.sync_results_every,
    )
