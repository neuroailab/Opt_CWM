import random

import torch
from torch import nn

from utils import utils


class PointSampler(nn.Module):
    def __init__(self, num_points):
        self.num_points = num_points

        super().__init__()

    def forward(self, b, h, w):
        pts = self._sample_pts(b, h, w)
        assert pts.size() == (b, self.num_points, 2)

        return pts

    def _sample_pts(self, b, h, w):
        raise NotImplementedError()


class UniformRandomSampler(PointSampler):
    def _sample_pts(self, b, h, w):
        grid = utils.gen_grid(0, h - 1, 0, w - 1, h, w)  # (HW), 2
        inds = torch.LongTensor([random.sample(range(h * w), self.num_points) for _ in range(b)])  # B, npts

        return grid[inds]  # B, npts, 2
