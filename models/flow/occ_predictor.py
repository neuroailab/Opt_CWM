from typing import List

import torch
from torch import nn


class OccPredictor(nn.Module):
    def __init__(self, spatial_reduction, masking_reduction, thresh):
        """
        Predicts occlusion from delta images.

        Arguments:
          spatial_reduction: Reduction across a single delta image.
          masking_reduction: Reduction across multiple delta images,
            each generated from a different random mask (multi-mask).
          thresh: Threshold to determine occlusion.
        """
        super().__init__()
        self.spatial_reduction = spatial_reduction
        self.masking_reduction = masking_reduction
        self.thresh = thresh

    def _compute_occ_metric(self, reduced_delt, masking_reduction="mean", spatial_reduction="mean"):
        # reduced_delt: M, B, H, W

        if spatial_reduction == "max":
            per_mask_metric = torch.amax(reduced_delt, dim=(2, 3))  # M, B
        elif spatial_reduction == "mean":
            per_mask_metric = torch.mean(reduced_delt, dim=(2, 3))  #  M, B
        else:
            raise ValueError(f"Invalid spatial reduction: {spatial_reduction}")

        if masking_reduction == "max":
            metric = torch.amax(per_mask_metric, dim=0)
        elif masking_reduction == "min":
            metric = torch.amin(per_mask_metric, dim=0)
        elif masking_reduction == "mean":
            metric = torch.mean(per_mask_metric, dim=0)
        elif masking_reduction == "median":
            metric = torch.median(per_mask_metric, dim=0)
        else:
            raise ValueError(f"Invalid masking reduction: {masking_reduction}")

        return metric  # B,

    def forward(self, recon_deltas: List[torch.Tensor]):
        """
        Predicts occlusion.

        Arguments:
          recon_deltas: List of predicted frame1 deltas.

        Returns:
          (torch.Tensor): True if occluded, False otherwise.
        """
        all_recon_deltas = torch.stack(recon_deltas, 0)  # M, B, C=3, H, W

        # M, B, C=3, H, W -> M, B, H, W
        all_reduced_recon_deltas = torch.norm(all_recon_deltas, p=1, dim=2)

        batch_occ_metric = self._compute_occ_metric(
            all_reduced_recon_deltas, masking_reduction=self.masking_reduction, spatial_reduction=self.spatial_reduction
        )  # (B,)

        pred_occlusion = batch_occ_metric < self.thresh
        return pred_occlusion
