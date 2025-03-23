"""Learning a flow 'pushforward': given an RGB frame and the full flow computed from
somewhere, predict the next frame of RGB.
"""

from abc import ABC
from pathlib import Path
from typing import Annotated

import torch
import torch.nn as nn
from torch import Tensor

from models.two_stream_cwm import PathLike, VideoDims, VideoTensor
from models.two_stream_cwm.masking_generator import FrameGroup, HeterogeneousFrameMasker
from models.two_stream_cwm.models.two_stream_vmae.two_stream_vmae import TwoStreamMaskingVMAE
from utils.utils import imagenet_normalize

FlowTensor = Annotated[Tensor, ["batch", "2", "1", "height", "width"]]


class FlowFwd(ABC, nn.Module):
    """FlowFwd
    Primary stream: RGB0
    Secondary stream: RGB0 \cat Flow
    """

    def __init__(
        self,
        two_stream_vmae: TwoStreamMaskingVMAE,
        device: str,
        flow_mask_fraction: float = 0.99,
    ):
        super().__init__()
        self.preprocess = imagenet_normalize
        self.two_stream_vmae = two_stream_vmae.to(device)
        self.device = device

        primary_mask_size = two_stream_vmae.encoder.mask_size_primary
        secondary_mask_size = two_stream_vmae.encoder.mask_size_secondary

        self.primary_masker = HeterogeneousFrameMasker(
            height=primary_mask_size[-2],
            width=primary_mask_size[-1],
            frame_groups=[FrameGroup(mask_fraction=0, num_frames=1)],
        )

        self.secondary_masker = HeterogeneousFrameMasker(
            height=secondary_mask_size[-2],
            width=secondary_mask_size[-1],
            frame_groups=[FrameGroup(mask_fraction=flow_mask_fraction, num_frames=1)],
        )

    # @abstractmethod
    def compute_flow(self, videos: VideoTensor) -> FlowTensor:
        """Compute flow from a set of videos

        Args:
            videos: a (Batch, Channel, Time, Height, Width) tensor

        Returns:
            a (Batch, 2, Height, Width) tensor of pixelwise flow
        """
        raise NotImplementedError

    def get_primary_stream_inputs(self, videos: VideoTensor) -> tuple[Tensor, Tensor]:
        """Primary stream just gets masked RGB"""
        videos = self.preprocess(videos)

        frame0 = videos[:, :, 0:1]
        masks = self.primary_masker(frame0)
        return frame0, masks

    def get_secondary_stream_inputs(self, videos: VideoTensor) -> tuple[Tensor, Tensor]:
        """Primary stream just gets masked RGB"""
        # compute flow before preprocessing
        flow = self.compute_flow(videos)

        # preprocess videos after flow has been computed
        preproc_rgb0 = self.preprocess(videos)[:, :, 0:1]

        x = torch.cat([preproc_rgb0, flow], dim=VideoDims.CHANNEL)
        return x, self.secondary_masker(x), flow

    def forward(self, videos: VideoTensor):
        x_primary, mask_primary = self.get_primary_stream_inputs(videos=videos)
        x_secondary, mask_secondary, flow = self.get_secondary_stream_inputs(videos=videos)

        out_primary, out_secondary = self.two_stream_vmae(
            x_primary=x_primary,
            mask_primary=mask_primary,
            x_secondary=x_secondary,
            mask_secondary=mask_secondary,
        )

        # return reconstruction, primary (all viz maks),
        # secondary (flow mask) and actual flow
        return out_primary, mask_primary, mask_secondary, flow

    def counterfactual_flow(
        self,
        videos: VideoTensor,
        flow: FlowTensor,
        flow_mask,  # Make the flow mask a necessary input (remove =None default) - DW
    ):
        x_primary, mask_primary = self.get_primary_stream_inputs(videos=videos)
        preproc_rgb0 = self.preprocess(videos)[:, :, 0:1]

        x_secondary = torch.cat([preproc_rgb0, flow], dim=VideoDims.CHANNEL)
        mask_secondary = self.secondary_masker(x_secondary) if flow_mask is None else flow_mask

        out_primary, out_secondary = self.two_stream_vmae(
            x_primary=x_primary,
            mask_primary=mask_primary,
            x_secondary=x_secondary,
            mask_secondary=mask_secondary,
        )

        # retain only the output of the primary (RGB) stream -- it'll be a prediction
        # of frame 1
        return out_primary, mask_primary

    def forward_primary(self, videos: VideoTensor):
        x, mask = self.get_primary_stream_inputs(videos=videos)
        shape = x.shape
        fake_secondary = torch.zeros(shape[0], shape[1] + 2, 1, *shape[-2:]).to(x)
        mask_secondary = self.secondary_masker(fake_secondary)

        out_primary, _ = self.two_stream_vmae(
            x_primary=x,
            mask_primary=mask,
            x_secondary=fake_secondary,
            mask_secondary=mask_secondary,
        )

        return out_primary, mask

    @classmethod
    def from_config(cls, cfg_path: PathLike, device: str = "cuda:0", flow_mask_fraction: float = 0.99):
        two_stream_vmae = TwoStreamMaskingVMAE.from_config_json(Path(cfg_path))
        return cls(
            two_stream_vmae=two_stream_vmae,
            device=device,
            flow_mask_fraction=flow_mask_fraction,
        )
