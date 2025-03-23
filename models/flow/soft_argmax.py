import torch
from torch import nn

from utils import utils


class SoftArgmax(nn.Module):
    def __init__(self, reduce_fn="l1", inv_temp=50):
        """
        Given a 2d delta image, computes differential argmax based on expectation.

        Arguments:
          reduce_fn: Channel reduction method of the delta image.
          inv_temp: Inverse temperature scaling.
        """
        super().__init__()

        self.reduce_fn = {  # Assumes input of form B, C, H, W
            "l1": lambda x: torch.norm(x, p=1, dim=1),
            "l2": lambda x: torch.norm(x, p=2, dim=1),
        }[reduce_fn]

        self.inv_temp = inv_temp

    def _compute_2d_expectation(self, prob_dist, H, W):
        prob_dist = prob_dist.reshape(H * W, 1)  # Add dimension corresponding to (row,col) of grid
        assert torch.allclose(prob_dist.sum(), torch.tensor(1.0), rtol=1e-4), prob_dist.sum().item()

        coord_grid = utils.gen_grid(0, H - 1, 0, W - 1, H, W).to(prob_dist.device)
        assert coord_grid.shape == (H * W, 2)

        expected_coord = torch.sum(coord_grid * prob_dist, dim=0)  # (2,)
        # print("Expected coord shape,value:", expected_coord.shape, expected_coord)

        diffs = coord_grid - expected_coord
        sq_dists = (diffs**2).sum(dim=-1, keepdim=True)  # (H*W,)
        var = (sq_dists * prob_dist).sum()  # (,)
        std = var.sqrt()

        return expected_coord, std

    def forward(self, delta_image: torch.Tensor):
        """
        Computes differential argmax on 2d grid.

        Arguments:
          delta_image: Delta image of shape B,C=3,H,W

        Returns:
          (torch.Tensor): The expected location (yx).
          (torch.Tensor): The standard deviation.
          (torch.Tensor): Temperature-scaled heatmap from deltas.
          (torch.Tensor): The channel-reduced delta image.
        """

        B, C, H, W = delta_image.shape
        assert C == 3, "Softargmax expects 3 channels"

        # Reduction across channels
        reduced_delta_image = self.reduce_fn(delta_image)
        assert reduced_delta_image.shape == (B, H, W)

        if self.inv_temp != -1:
            scaled_delta_image = self.inv_temp * reduced_delta_image  # softmax temperature

        locs = []
        stds = []
        prob_heatmaps = []

        for b in range(B):

            if self.inv_temp == -1:
                prob_dist = reduced_delta_image[b].flatten() / reduced_delta_image[b].sum()
            elif self.inv_temp == -2:
                red_delta_img_bf = reduced_delta_image[b].flatten()
                red_delta_img_bf_sq = red_delta_img_bf**2
                prob_dist = red_delta_img_bf_sq / red_delta_img_bf_sq.sum()
            else:
                assert self.inv_temp > 0
                logits = scaled_delta_image[b]
                assert logits.shape == (H, W)
                prob_dist = nn.functional.softmax(logits.flatten(), dim=0)

            assert prob_dist.shape == (H * W,)
            assert torch.allclose(prob_dist.sum(), torch.tensor(1.0), rtol=1e-4), prob_dist.sum().item()

            loc, std = self._compute_2d_expectation(prob_dist, H, W)
            locs.append(loc)
            stds.append(std)
            prob_heatmaps.append(prob_dist.reshape(H, W))

        loc = torch.stack(locs, dim=0)
        std = torch.stack(stds, dim=0)
        prob_heatmap = torch.stack(prob_heatmaps, dim=0)

        return loc, std, prob_heatmap, reduced_delta_image
