import os

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from external.sea_raft import utils as raft_utils
from external.sea_raft.core import raft
from utils import constants, dist_logging, flow_utils, utils

DEFAULT_CFG_PATH = "external/sea_raft/config/distilled_cfg.json"

logger = dist_logging.get_logger(__name__)


class RAFTWrapper(nn.Module):
    def __init__(self, path_to_cfg=None):
        super().__init__()
        self.path_to_cfg = path_to_cfg or DEFAULT_CFG_PATH
        self.args = raft_utils.json_to_args(self.path_to_cfg)
        self.raft = raft.RAFT(self.args)

    def _compute_flow_map(self, frame0, frame1):
        output = self.raft(frame0, frame1, iters=self.args.iters, test_mode=True)
        flow = output["flow"][-1]

        flow = F.interpolate(flow, scale_factor=0.5**self.args.scale, mode="bilinear", align_corners=False) * (
            0.5**self.args.scale
        )

        return flow

    def _get_cycle_consistency_map(self, flow_fwd, flow_bck, occ_thresh=0.5, how="rel"):
        fwd_bck_cycle, _ = flow_utils.backward_warp(img2=flow_bck, flow=flow_fwd)
        flow_diff_fwd = flow_fwd + fwd_bck_cycle

        bck_fwd_cycle, _ = flow_utils.backward_warp(img2=flow_fwd, flow=flow_bck)
        flow_diff_bck = flow_bck + bck_fwd_cycle

        if how == "rel":
            # This was the function as previously written

            norm_fwd = flow_utils.l2_norm(flow_fwd) ** 2 + flow_utils.l2_norm(fwd_bck_cycle) ** 2
            norm_bck = flow_utils.l2_norm(flow_bck) ** 2 + flow_utils.l2_norm(bck_fwd_cycle) ** 2

            occ_thresh_fwd = occ_thresh * norm_fwd + 0.5
            occ_thresh_bck = occ_thresh * norm_bck + 0.5

            occ_mask_fwd = 1 - (flow_utils.l2_norm(flow_diff_fwd) ** 2 > occ_thresh_fwd).float()
            occ_mask_bck = 1 - (flow_utils.l2_norm(flow_diff_bck) ** 2 > occ_thresh_bck).float()

        elif how == "abs":
            # Modified to use absolute pixel threshold
            occ_mask_fwd = 1 - (flow_utils.l2_norm(flow_diff_fwd) > occ_thresh).float()
            occ_mask_bck = 1 - (flow_utils.l2_norm(flow_diff_bck) > occ_thresh).float()

        else:
            raise ValueError(f"Invalid thresholding method: {how=}")

        return occ_mask_fwd

    def forward(self, video, pixel_loc0, return_flow_map=True):
        video = utils.imagenet_unnormalize(video)
        video = video * 255.0

        f0, f1 = torch.unbind(video, 2)
        f0 = F.interpolate(f0, scale_factor=2**self.args.scale, mode="bilinear", align_corners=False)
        f1 = F.interpolate(f1, scale_factor=2**self.args.scale, mode="bilinear", align_corners=False)

        flow_fwd = self._compute_flow_map(f0, f1)
        flow_bck = self._compute_flow_map(f1, f0)

        occ_mask = self._get_cycle_consistency_map(flow_fwd, flow_bck, 6, "abs")

        B, C, H, W = occ_mask.size()
        assert C == 1 and (H, W) == video.size()[-2:]
        occ_mask = occ_mask.squeeze(1)

        indices = torch.arange(B).long()

        pixel_row = pixel_loc0[:, 0]
        pixel_col = pixel_loc0[:, 1]
        pred_occs = 1 - occ_mask[indices, pixel_row, pixel_col]  # B,

        diffs = flow_fwd[indices, :, pixel_row, pixel_col].squeeze(-1)
        utils.size_guard(diffs, (B, 2))

        pred_loc = pixel_loc0 + diffs.flip(-1)

        result = {
            "expec_pred_pixel_loc": pred_loc,
            "argmax_pred_pixel_loc": torch.zeros_like(pred_loc),
            "multi_scale_pred_pixel_loc": torch.zeros_like(pred_loc),
            "pred_pixel_loc": pred_loc,
            "pred_occ": pred_occs,
        }

        if return_flow_map:
            flow_maps = []
            flow_fwd_viz = flow_fwd.float().cpu().permute(0, 2, 3, 1).detach().numpy()
            for b in range(video.size(0)):
                flow_maps.append(flow_utils.flow_to_image(flow_fwd_viz[b]))

            result["flow_maps"] = np.stack(flow_maps, 0)
            result["flow_fwd"] = flow_fwd

        return result

    def load_pretrained(self, force=False):
        local_dir = os.path.join(constants.MODEL_LOCAL_CACHE_PATH, "opt_cwm", "distilled_sea_raft")
        os.makedirs(local_dir, exist_ok=True)

        gcloud_dir = os.path.join(constants.MODEL_GCLOUD_BUCKET_PATH, "opt_cwm", "distilled_sea_raft")

        local_path = os.path.join(local_dir, "ckpt.pt")
        gcloud_path = os.path.join(gcloud_dir, "ckpt.pt")

        if force or not os.path.exists(local_path):
            logger.info(f"Saving distilled_sea_raft to: {local_path}.")
            utils.download_from_gcloud(gcloud_path, local_path)

        ckpt = torch.load(local_path, map_location="cpu")
        self.raft.load_state_dict(ckpt)

        logger.info("Succesfully loaded checkpoint for distilled_sea_raft.")

        return self
