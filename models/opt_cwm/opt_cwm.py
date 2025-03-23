import os

import torch
from torch import nn

from models.flow import flow_predictor
from models.opt_cwm import point_sampler
from models.two_stream_cwm.models.flowcontrol import flow_fwd
from utils import constants, dist_logging, utils

logger = dist_logging.get_logger(__name__)


class OptCWM(nn.Module):
    def __init__(
        self,
        flow_predictor: flow_predictor.FlowPredictor,
        recon_cwm_model: flow_fwd.FlowFwd,
        n_flow_pts: int,
    ):
        """
        Uses TwoStreamCWM to predict next frame based on frame0 RGB and sparse flow.
        Sparse flow is predicted from the FlowPredictor. This allows for
        jointly optimizing the FlowPredictor in an unsupervised way.

        Arguments:
          flow_predictor: The FlowPredictor model used to generate sparse flow.
          recon_cwm_model: The TwoStreamCWM model that reconstructs the next frame.
          n_flow_pts: Number of flow points to sample for sparse flow map.
        """
        super().__init__()

        self.flow_predictor = flow_predictor
        self.recon_cwm_model = recon_cwm_model
        self.n_flow_pts = n_flow_pts

        self.input_size = recon_cwm_model.two_stream_vmae.encoder.input_size_primary[-2:]
        self.patch_size = recon_cwm_model.two_stream_vmae.encoder.params.primary.patch_size[-1]
        self.n_patches = [inp // self.patch_size for inp in self.input_size]

        self.point_sampler = point_sampler.UniformRandomSampler(self.n_flow_pts)

    def _compute_sparse_flow(self, video):
        B, C, T, H, W = video.size()

        flow_patch_indices = self.point_sampler(B, *self.n_patches).to(video.device).long()
        # pick center pixel of patch
        flow_pixel_indices = flow_patch_indices * self.patch_size + self.patch_size // 2

        pred_loc = torch.stack(
            [
                self.flow_predictor(video, flow_pixel_indices[:, i])["expec_pred_pixel_loc"]
                for i in range(self.n_flow_pts)
            ],
            dim=1,
        ).to(
            flow_pixel_indices.device
        )  # (B, N, 2)

        sparse_flow = pred_loc - flow_pixel_indices  # b, npts, 2

        flow_mask = torch.ones((B, *self.n_patches)).bool().to(video.device)

        row, col = torch.unbind(flow_patch_indices, -1)
        batch_indices = torch.arange(B).unsqueeze(-1).repeat_interleave(self.n_flow_pts, -1)

        flow_mask[batch_indices, row, col] = False

        flow_map = torch.zeros((B, 2, *self.n_patches)).to(video.device)  # b, npts, 2
        flow_map[batch_indices, :, row, col] = torch.roll(sparse_flow.float(), 1, -1)  # yx -> xy

        flow_map = flow_map.unsqueeze(2)  # B, 2, 1, h, w
        flow_map = flow_map.repeat_interleave(self.patch_size, -1).repeat_interleave(self.patch_size, -2)

        return flow_map, flow_mask.flatten(start_dim=1), flow_patch_indices

    def forward(self, video: torch.Tensor):
        """
        Generates reconstructed next frame.

        Arguments:
          video: The video to predict flow, shape B,C=3,T=2,H,W.

        Returns:
          (torch.Tensor): Reconstructed frame1.
        """
        sparse_flow, flow_mask, flow_patch_indices = self._compute_sparse_flow(video)

        frame1_recon, _ = self.recon_cwm_model.counterfactual_flow(video, sparse_flow, flow_mask)

        return frame1_recon

    def load_pretrained(self, force=False):
        self.flow_predictor.cwm_model.load_pretrained(force=force)

        local_dir = os.path.join(constants.MODEL_LOCAL_CACHE_PATH, "opt_cwm")
        os.makedirs(local_dir, exist_ok=True)

        gcloud_dir = os.path.join(constants.MODEL_GCLOUD_BUCKET_PATH, "opt_cwm")

        opt_cwm_local_path = os.path.join(local_dir, "opt_cwm_ckpt.pt")
        opt_cwm_gcloud_path = os.path.join(gcloud_dir, "opt_cwm_ckpt.pt")

        if force or not os.path.exists(opt_cwm_local_path):
            logger.info(f"Saving opt_cwm model to: {opt_cwm_local_path}.")
            utils.download_from_gcloud(opt_cwm_gcloud_path, opt_cwm_local_path)

        ckpt = torch.load(opt_cwm_local_path, map_location="cpu")
        self.load_state_dict(ckpt["model"], strict=False)

        logger.info("Succesfully loaded checkpoint for opt_cwm.")

        return self


if __name__ == "__main__":
    B, C, T, H, W = 2, 3, 2, 16, 16
    mdl = OptCWM(None, None, 5, frame_size=H)
    mdl._compute_sparse_flow(torch.zeros((B, C, T, H, W)), None, None, None)
