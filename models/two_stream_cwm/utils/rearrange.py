from einops import rearrange
from torch import Tensor

from models.two_stream_cwm import ThreeTuple, VideoTensor


def flip_temporal_and_channel_dims(videos: VideoTensor) -> VideoTensor:
    return rearrange(videos, "B T C H W -> B C T H W")


def unpatchify(tokens: Tensor, num_channels: int, patch_size: ThreeTuple) -> VideoTensor:
    t, h, w = patch_size
    try:
        return rearrange(
            tokens,
            "B N (t h w c) -> (B N) c t h w",
            c=num_channels,
            t=t,
            h=h,
            w=w,
        )
    except Exception:
        raise ValueError("tokens doesn't have valid shape: %s" % tokens.shape)


def patchify(video: Tensor, patch_size: ThreeTuple) -> VideoTensor:
    B, C, T, H, W = video.shape
    pT, pH, pW = patch_size
    nT = T // pT
    nH = H // pH
    nW = W // pW

    try:
        return rearrange(
            video,
            "B C (nT pT) (nH pH) (nW pW) -> B C (nT nH nW) (pT pH pW)",
            nT=nT,
            nH=nH,
            nW=nW,
            pT=pT,
            pH=pH,
            pW=pW,
            B=B,
            C=C,
        )
    except Exception:
        raise ValueError("video doesn't have valid shape: %s" % video.shape)
