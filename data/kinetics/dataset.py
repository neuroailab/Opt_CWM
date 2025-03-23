import os

import decord
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms

from data.kinetics import dataset_utils


class VideoMAE(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    transform : function, default None.
        A function that takes data and label and transforms them.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    """

    def __init__(
        self,
        root,
        setting,
        train=True,
        video_ext="mp4",
        num_segments=1,
        new_length=1,
        new_step=1,
        randomize_interframes=False,
        transform=None,
        temporal_jitter=False,
        video_loader=False,
    ):

        super(VideoMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.num_segments = num_segments
        self.new_length = new_length

        self.randomize_interframes = randomize_interframes
        self._new_step = new_step
        self.temporal_jitter = temporal_jitter
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.transform = transform

        self.clips = self._make_dataset(root, setting)

        if len(self.clips) == 0:
            raise (
                RuntimeError(
                    "Found 0 video clips in subfolders of: " + root + "\n" + "Check your data directory (opt.data-dir)."
                )
            )

    def __getitem__(self, index):

        directory, target = self.clips[index]

        if self.video_loader:
            if "." in directory.split("/")[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
            else:
                # data in the "setting" file do not have extension, e.g., demo
                # So we need to provide extension (i.e., .mp4) to complete the file name.
                video_name = "{}.{}".format(directory, self.video_ext)

            try:
                decord_vr = decord.VideoReader(video_name, num_threads=1)
            except:
                # return video_name
                return self.__getitem__(index + 1)
            duration = len(decord_vr)

        segment_indices, skip_offsets, new_step, skip_length = self._sample_train_indices(duration)

        images = self._video_TSN_decord_batch_loader(
            directory, decord_vr, duration, segment_indices, skip_offsets, new_step, skip_length
        )

        process_data, mask = self.transform((images, None))
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0, 1)

        return (process_data, mask)

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting):
        if not os.path.exists(setting):
            raise (RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(" ")
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise (RuntimeError("Video input format is not correct, missing one or more element. %s" % line))
                elif len(line_info) > 2:
                    line_info = (" ".join(line_info[:-1]), line_info[-1])  # filename has spaces
                clip_path = os.path.join(line_info[0])
                target = int(line_info[1])
                item = (clip_path, target)
                clips.append(item)
        # import torch_xla.core.xla_model as xm
        # print = xm.master_print
        # print("Dataset created. Number of clips: ", len(clips))
        return clips

    def _sample_train_indices(self, num_frames):
        if self.randomize_interframes is False:
            new_step = self._new_step
        else:
            new_step = np.random.randint(1, self._new_step + 1)

        skip_length = self.new_length * new_step

        average_duration = (num_frames - skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(average_duration, size=self.num_segments)
        elif num_frames > max(self.num_segments, skip_length):
            offsets = np.sort(np.random.randint(num_frames - skip_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(new_step, size=skip_length // new_step)
        else:
            skip_offsets = np.zeros(skip_length // new_step, dtype=int)
        return offsets + 1, skip_offsets, new_step, skip_length

    def _video_TSN_decord_batch_loader(
        self, directory, video_reader, duration, indices, skip_offsets, new_step, skip_length
    ):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, skip_length, new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + new_step < duration:
                    offset += new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [
                Image.fromarray(video_data[vid, :, :, :]).convert("RGB") for vid, _ in enumerate(frame_id_list)
            ]
        except:
            raise RuntimeError(
                "Error occured in reading frames {} from video {} of duration {}.".format(
                    frame_id_list, directory, duration
                )
            )
        return sampled_list


class ContextAndTargetVideoDataset(VideoMAE):
    """
    A video dataset whose provided videos consist of (1) a "context" sequence of length Tc
    and (2) a "target" sequence Tt.

    These two sequences have the same frame rate (specificiable in real units) but are
    separated by a specified gap (which may vary for different examples.)

    The main use case is for training models to predict ahead by some variable amount,
    given the context.
    """

    standard_fps = [12, 24, 30, 48, 60, 100]

    def __init__(
        self,
        root,
        setting,
        train=True,
        transform=None,
        step_units="ms",
        new_step=150,
        start_frame=0,
        context_length=2,
        target_length=1,
        channels_first=True,
        context_target_gap=[400, 600],
        normalize_timestamps=True,
        default_fps=30,
        min_fps=0.1,
        seed=0,
        *args,
        **kwargs
    ):
        super(ContextAndTargetVideoDataset, self).__init__(
            root=root,
            setting=setting,
            train=train,
            transform=transform,
            new_length=context_length,
            video_loader=True,
            *args,
            **kwargs
        )

        # breakpoint()

        self.context_length = self.new_length
        self.target_length = target_length

        ## convert from fps and step size to frames
        self._fps = None
        self._min_fps = min_fps
        self._default_fps = default_fps
        self._step_units = step_units
        self.new_step = new_step

        ## sampling for train and test
        self._start_frame = start_frame
        self.gap = context_target_gap
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)

        # breakpoint()

        ## output formatting
        self._channels_first = channels_first
        self._normalize_timestamps = normalize_timestamps

    def _get_frames_per_t(self, t):
        if self._step_units == "frames" or (self._step_units is None):
            return int(t)

        assert self._fps is not None
        t_per_frame = 1 / self._fps
        if self._step_units in ["ms", "milliseconds"]:
            t_per_frame *= 1000.0

        return max(int(np.round(t / t_per_frame)), 1)

    @property
    def new_step(self):
        if self._fps is None:
            return None
        else:
            return self._get_frames_per_t(self._new_step)

    @new_step.setter
    def new_step(self, v):
        self._new_step = v

    @property
    def gap(self):
        if self._fps is None:
            return [1, 2]
        else:
            gap = [self._get_frames_per_t(self._gap[0]), self._get_frames_per_t(self._gap[1])]
            gap[1] = max(gap[1], gap[0] + 1)
            return gap

    @gap.setter
    def gap(self, v):
        if v is None:
            v = self._new_step
        if not isinstance(v, (list, tuple)):
            v = [v, v]
        self._gap = v

    def _get_video_name(self, directory):
        if "".join([".", self.video_ext]) in directory.split("/")[-1]:
            # data in the "setting" file has extension, e.g. demo.mpr
            video_name = directory
        else:
            # data doesn't have an extension
            video_name = "{}.{}".format(directory, self.video_ext)
        return video_name

    def _set_fps(self, reader):
        """click fps to a standard"""
        if self._step_units == "frames" or self._step_units is None:
            self._fps = None
        else:
            self._fps = None
            fps = reader.get_avg_fps()
            for st in self.standard_fps:
                if (int(np.floor(fps)) == st) or (int(np.ceil(fps)) == st):
                    self._fps = st
            if self._fps is None:
                self._fps = int(np.round(fps))

            if self._fps < self._min_fps:
                self._fps = self._default_fps

    def _get_step_and_gap(self):
        step = self.new_step
        if self.randomize_interframes and self.train:
            step = self.rng.randint(1, step + 1)

        if self.train:
            gap = self.rng.randint(*self.gap)
        else:
            gap = sum(self.gap) // 2
        return (step, gap)

    def _sample_frames(self):
        step, gap = self._get_step_and_gap()

        ## compute total length of sample
        ## e.g. if context_length = 2, step = 1, gap = 10, target_length = 2:
        ## total_length = 2 * 1 + 10 + (2 - 1) * 1 = 13
        ## so len(video) must be >= 13
        self._total_length = self.context_length * step + gap + (self.target_length - 1) * step
        if self._total_length > (self._num_frames - self._start_frame):
            if self.train:
                return None
            else:
                raise ValueError(
                    "movie of length %d starting at fr=%d is too long for video of %d frames"
                    % (self._total_length, self._start_frame, self._num_frames)
                )

        ## sample the frames randomly (if training) or from the start frame (if test)
        if self.train:
            self.start_frame_now = self.rng.randint(
                min(self._start_frame, self._num_frames - self._total_length), self._num_frames - self._total_length + 1
            )
        else:
            self.start_frame_now = min(self._start_frame, self._num_frames - self._total_length)

        frames = [self.start_frame_now + i * step for i in range(self.context_length)]
        frames += [frames[-1] + gap + i * step for i in range(self.target_length)]

        # breakpoint()

        return frames

    def _decode_frame_images(self, reader, frames):
        try:
            video_data = reader.get_batch(frames).asnumpy()
            video_data = [Image.fromarray(video_data[t, :, :, :]).convert("RGB") for t, _ in enumerate(frames)]
        except:
            raise RuntimeError(
                "Error occurred in reading frames {} from video {} of duration {}".format(
                    frames, self.index, self._num_frames
                )
            )
        return video_data

    def __getitem__(self, index):

        self.index = index
        self.directory, target = self.clips[index]

        self.video_name = self._get_video_name(self.directory)

        ## build decord loader
        try:
            decord_vr = decord.VideoReader(self.video_name, num_threads=1)
            self._set_fps(decord_vr)
        except:
            # return self.video_name
            return self.__getitem__(index + 1)

        ## sample the video
        self._num_frames = len(decord_vr)
        self.frames = self._sample_frames()
        if self.frames is None:
            print("no movie of length %d for video idx=%d" % (self._total_length, self.index))
            return self.__getitem__(index + 1)

        ## decode to PIL.Image
        image_list = self._decode_frame_images(decord_vr, self.frames)

        ## postproc to torch.Tensor and mask generation
        if self.transform is None:
            image_tensor = torch.stack([transforms.ToTensor()(img) for img in image_list], 0)
        else:
            image_tensor = self.transform((image_list, None))

            image_tensor = image_tensor.view(self.context_length + self.target_length, 3, *image_tensor.shape[-2:])

        ## VMAE expects [B,C,T,H,W] rather than [B,T,C,H,W]
        if self._channels_first:
            image_tensor = image_tensor.transpose(0, 1)

        return image_tensor


def get_kinetics_loader(
    augmentation_type,
    crop_size,
    augmentation_scales,
    data_path,
    frame_delta_ms,
    samples_per_epoch,
    batch_size,
    num_workers=4,
):

    # dataset_list = []
    data_transform = dataset_utils.DataAugmentationForVideoMAE(augmentation_type, crop_size, augmentation_scales)

    # for data_path in [args.data_path] if args.data_path_list is None else args.data_path_list:
    dataset = ContextAndTargetVideoDataset(
        root=None,
        setting=data_path,
        video_ext="mp4",
        context_length=1,
        target_length=1,
        step_units="ms",
        new_step=frame_delta_ms,
        context_target_gap=frame_delta_ms,
        transform=data_transform,
        randomize_interframes=False,
        channels_first=True,
        temporal_jitter=False,
        train=True,
    )

    train_sampler = RandomSampler(dataset, replacement=True, num_samples=samples_per_epoch)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
