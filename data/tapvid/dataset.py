import pickle
from dataclasses import dataclass
from typing import Any, List, Literal, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

from data.tapvid import dataset_utils
from utils import constants, dist_logging, utils

logger = dist_logging.get_logger(__name__)


@dataclass(frozen=True)
class VideoSpec:
    vidname: str
    num_query_pts: int
    num_target_frames: int
    num_expected_evals: int
    gt_pts: np.ndarray
    gt_occ: np.ndarray
    dataset_res: tuple


class FlatTAPVidDataset(Dataset):
    def __init__(
        self,
        resolution: List[int] | Literal["native", "nearest_16"],
        frame_delta=-1,
        path_to_pkl="data/tapvid_davis.pkl",
        debug=False,
    ):
        """
        Abstract class for TAP-Vid evaluation.
        Flattens the dataset such that each item is a single point
        for a single frame pair, allowing for increased parallelism.

        Arguments:
          resolution: Resolution of video returned by dataset. Either
            explicitly pass in [H, W], or choose between the two pre-defined flags.
          frame_delta: Frame gap between pairs. If <0, dataset includes all frame pairs
            starting from the first visible frame ("first"). Otherwise, runs
            constant frame gap ("cfg") evaluation with the given delta.
          path_to_pkl: Path to TAP-Vid pkl file.
          debug: If true, run on 3 videos.
        """

        self.frame_delta = frame_delta
        self.resolution = resolution

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(constants.IMAGENET_DEFAULT_MEAN, constants.IMAGENET_DEFAULT_STD),
            ]
        )

        self.video_names, self.data = self._extract_data_from_pkl(path_to_pkl)

        if debug:
            self.video_names = self.video_names[:3]
            self.data = {k: v for k, v in self.data.items() if k in self.video_names}

            logger.info(f"Running TAPVid Dataset with Debug mode: {self.video_names}")

        if self.frame_delta < 0:
            logger.info("TAPVid Evaluation Mode First")
            self.query_points, self.vid_specs = self._process_mode_first()
        else:
            logger.info(f"TAPVid Evaluation Mode CFG-{self.frame_delta}")
            self.query_points, self.vid_specs = self._process_mode_cfg()

    def __len__(self):
        return len(self.query_points)

    def _extract_data_from_pkl(self, path_to_pkl):
        raise NotImplementedError()

    def _frames_to_video(self, frames):
        raise NotImplementedError()

    def _calc_resolution(self, h, w):
        if self.resolution == "nearest_16":
            h = (h // 16) * 16
            w = (w // 16) * 16
        elif not isinstance(self.resolution, str):
            h, w = self.resolution

        return h, w

    def _process_mode_cfg(self) -> Tuple[List[Any], List[VideoSpec]]:
        """
        Flatten TAP-Vid Dataset for CFG evaluation.

        Returns:
          (list(np.array)): List of datapoints, each of which is a 4-tuple
            (video_idx, point_idx, start_time, end_time). This information can
            uniquely retrieve the right data.
          (list(VideoSpec)): List of VideoSpec for each video.
            This VideoSpec helps collate the flattened datapoints for evaluation.
        """
        all_query_points = []
        all_video_specs = []

        for i, vidname in enumerate(self.video_names):
            _data = self.data[vidname]
            points = _data["points"]  # N, T, 2(x,y)

            occ = _data["occluded"]

            video_queries = []
            for pt in range(points.shape[0]):
                # get all the valid starting point indices
                # starting point should not be occluded, and be able to extend delta into time
                viz_times = np.nonzero(~occ[pt])[0]
                viz_times = viz_times[viz_times < points.shape[1] - self.frame_delta]

                video_indices = np.full(viz_times.shape, i)
                point_indices = np.full(viz_times.shape, pt)

                # queries: (viz_times, 4) with 4-tuple (vi, pi, st, et)
                queries = np.stack([video_indices, point_indices, viz_times, viz_times + self.frame_delta], -1)

                video_queries.append(queries)

            all_query_points.extend(np.concatenate(video_queries, 0).tolist())

            video_queries = np.concatenate(video_queries, 0)

            # For CFG, we fix two target frames per point: (f0, f0) and (f0, f0 + delta)
            spec = VideoSpec(
                vidname=vidname,
                num_query_pts=video_queries.shape[0],
                num_target_frames=2,
                num_expected_evals=video_queries.shape[0],
                gt_pts=_data["points"],
                gt_occ=_data["occluded"],
                dataset_res=self._calc_resolution(*_data["resolution"]),
            )

            all_video_specs.append(spec)

        return all_query_points, all_video_specs

    def _process_mode_first(self) -> Tuple[List[Any], List[VideoSpec]]:
        """
        Flatten TAP-Vid Dataset for First evaluation.

        Returns:
          (list(np.array)): List of datapoints, each of which is a 4-tuple
            (video_idx, point_idx, start_time, end_time). This information can
            uniquely retrieve the right data.
          (list(VideoSpec)): List of VideoSpec for each video.
            This VideoSpec helps collate the flattened datapoints for evaluation.
        """
        all_query_points = []
        all_video_specs = []

        for i, vidname in enumerate(self.video_names):
            _data = self.data[vidname]

            occ = _data["occluded"]  # N, T
            T = occ.shape[-1]
            valid = np.sum(~occ, axis=1) > 0  # must be visible at some point

            valid_occ = occ[valid]  # Nv, T
            viz_points, viz_times = np.nonzero(~valid_occ)

            # for each point, get the first frame in which it is visible
            pi, ind = np.unique(viz_points, return_index=True)
            sti = viz_times[ind]

            vi = np.full(pi.shape, i)  # N,v

            query_points = np.vstack([vi, pi, sti])

            # e.g. for frame0, make it into [0, 0], [0, 1], ... , [0, T - 1]
            point_pairs = np.repeat(query_points, T, axis=1)
            point_pairs = np.vstack([point_pairs, np.concatenate([np.arange(T) for t in sti])])

            # each entry contains the 4-tuple (vi, pi, st, et)
            # st is fixed to first visible frame, et ranges from 0 to T
            all_query_points.extend(point_pairs.T.tolist())

            spec = VideoSpec(
                vidname=vidname,
                num_query_pts=valid_occ.shape[0],
                num_target_frames=T,
                num_expected_evals=point_pairs.shape[-1],
                gt_pts=_data["points"],
                gt_occ=_data["occluded"],
                dataset_res=self._calc_resolution(*_data["resolution"]),
            )

            all_video_specs.append(spec)

        return all_query_points, all_video_specs

    def __getitem__(self, idx):

        vi, pi, st, et = self.query_points[idx]

        vidname = self.video_names[vi]

        _data = self.data[vidname]

        video = self._frames_to_video(_data["video"][[st, et]])

        h, w = video.size()[-2:]
        rh, rw = self._calc_resolution(h, w)

        if (h, w) != (rh, rw):
            video = torch.nn.functional.interpolate(video, size=(rh, rw), mode="bilinear")

        pts = np.flip(_data["points"][pi, [st, et]], -1)  # T=2, 2(y,x)
        # rescale from raster to resized video scale
        pts = utils.rescale_points(torch.Tensor(pts.copy()), (1, 1), video.size()[-2:])

        return {
            "videos": video,  # T=2, C=3, H, W
            "points": pts,  # T=2, 2(y, x)
            "video_name": vidname,  # (1,)
            "occluded": _data["occluded"][pi, et],  # (1,)
            "uid": torch.Tensor([vi, pi, st, et]).long(),  # this should uniquely map where the prediction comes from,
        }


class FlatTAPVidDAVIS(FlatTAPVidDataset):
    def _extract_data_from_pkl(self, path_to_pkl):
        with open(path_to_pkl, "rb") as f:
            _data = pickle.load(f)

        for k in _data:
            _data[k]["resolution"] = _data[k]["video"].shape[1:3]

        video_names = sorted(list(_data))
        return video_names, _data

    def _frames_to_video(self, frames):
        return torch.stack([self.transform(frame) for frame in frames], 1)


class FlatTAPVidKinetics(FlatTAPVidDataset):
    def _extract_data_from_pkl(self, path_to_pkl):
        with open(path_to_pkl, "rb") as f:
            raw = pickle.load(f)
            if isinstance(raw, dict):
                raw = list(raw.values())

        _data = {}
        vidnames = []
        for i in range(len(raw)):
            vidname = raw[i]["video_id"]
            _data[vidname] = raw[i]
            _data[vidname]["resolution"] = dataset_utils.decode_byte_array_imgs(raw[i]["video"][0]).shape[:2]

            vidnames.append(vidname)

        return vidnames, _data

    def _frames_to_video(self, frames):
        frames = [dataset_utils.decode_byte_array_imgs(frame) for frame in frames]
        return torch.stack([self.transform(frame) for frame in frames], 1)


def get_tapvid_loader(dataset, resolution, frame_delta, path_to_pkl, batch_size, debug=False, num_workers=4):

    if dataset == "davis":
        dataset = FlatTAPVidDAVIS(resolution, frame_delta, path_to_pkl, debug)
    elif dataset == "kinetics":
        dataset = FlatTAPVidKinetics(resolution, frame_delta, path_to_pkl, debug)
    else:
        raise RuntimeError(f"Unrecognized TAP-Vid Dataset: {dataset}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
        drop_last=False,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=dataset_utils.collate_by_shape,
    )


if __name__ == "__main__":
    dataset = FlatTAPVidDAVIS("native", -1, "datasets/tapvid/tapvid_davis.pkl")
