"""Boolean mask generators."""

import itertools
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class FrameGroup:
    """Parameters for a group of frames that share a common mask ratio.

    Attributes:
        mask_fraction: the fraction of patches/tokens to be masked. Must be in [0, 1]
        num_frames: the number of frames masked at this fraction
    """

    mask_fraction: float
    num_frames: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        assert self.mask_fraction >= 0 and self.mask_fraction <= 1, "mask ratio out of bounds"


class MaskGenerator:
    """Mask generator base class.

    The core masking generator produces batches of masks of shape (B, N), where B
    is the batch size, and N is the total number of patches/tokens: T * H * W.
    """

    def __init__(
        self,
        input_size,
        mask_fraction: float = 0.9,
        seed: int | None = None,
    ):
        """
        Args:
            input_size: three-tuple specifying (num_frames, height, width)
            mask_fraction: fraction of patches to mask. Defaults to 0.9.
            seed: If provided, seeds the torch RNG when instantiated. Defaults to None.
        """
        super().__init__()
        assert len(input_size) == 3, "generating masks only for 3D (T, H, W) inputs"
        self._set_random_state(seed)
        self.mask_fraction = mask_fraction
        self.num_frames, self.height, self.width = input_size

        # determine number of visible and masked patches in each 2D frame
        self.num_patches = self.height * self.width
        self.num_masked = int(self.mask_fraction * self.num_patches)
        self.num_visible = self.num_patches - self.num_masked

    def __repr__(self) -> str:
        return (
            f"Class: {type(self).__name__}\n"
            f"Total patches per frame: {self.num_patches}\n"
            f"Masked patches per frame: {self.num_masked}\n"
            f"Mask Ratio: {self.mask_fraction:.03f}\n"
        )

    def set_num_visible(self, num_visible: int) -> None:
        self.num_visible = num_visible
        self.num_masked = self.num_patches - self.num_visible
        self.mask_fraction = self.num_visible / self.num_patches

    def set_mask_fraction(self, mask_fraction: float) -> None:
        self.mask_fraction = mask_fraction
        self.num_visible = int(self.num_patches * (1.0 - mask_fraction))
        self.num_masked = self.num_patches - self.num_visible

    def _set_random_state(self, seed: int | None = None) -> None:
        """Set the torch random state.

        Args:
            seed: RNG seed. Defaults to None, in which case 'fresh unpredictable
                entropy will be pulled from the OS'

        Raises:
            ve: If seed is provided but not an integer, raise ValueError
        """
        if seed is None:
            return

        try:
            torch.manual_seed(seed)
        except ValueError as ve:
            print("If provided, seed must be `int`")
            raise ve

    def sample_mask_per_frame(self):
        """Produce a single boolean mask.

        The mask is initialized as [False, False, ..., True, True], with as many Falses
          as there are visible patches, and as many Trues as there are masked patches.
          We then shuffle the mask.

        Returns:
            Shuffled 1-D mask
        """
        mask = torch.cat(
            [
                torch.zeros([self.num_visible]),
                torch.ones([self.num_masked]),
            ],
            0,
        ).bool()

        # get shuffled indices  and shuffle the mask
        inds = torch.randperm(mask.shape[0]).long()
        mask = mask[inds]

        return mask

    def __call__(self, x: Tensor | None = None, batch_size: int | None = None):
        """Generate a batch of masks. Batch size is inferred from `x`, if provided.
              Depending on the attr  `num_visibile_frames_to_prepend`, optionally add
              some number of empty frames at the front.

        Args:
            x: tensor from which to infer batch size. Defaults to None, in which case
              B = 1.

        Returns:
            (B, N) batch of masks, where N is the total number of tokens across frames,
            height, and width
        """
        if batch_size is None:
            batch_size = x.shape[0] if x is not None else 1

        def _all_masks():
            # produce a 1D mask per frame
            mask_per_frame = [self.sample_mask_per_frame() for _ in range(self.num_frames)]

            # concatenate all masks for each frame
            return torch.cat(mask_per_frame, dim=0)

        # batch the masks into a (B, N) Tensor
        batched_masks = torch.stack([_all_masks() for _ in range(batch_size)], dim=0)

        return batched_masks


class FuzzyMaskGenerator(MaskGenerator):
    """
    With probability p = fuzzy_prob, each revealed patch (value False)
    is replaced with a uniformly sampled float value in [0, 1].
    """

    def __init__(self, *args, fuzzy_prob: float = 0.25, **kwargs):
        super(FuzzyMaskGenerator, self).__init__(*args, **kwargs)
        self.fuzzy_prob = fuzzy_prob

    def __repr__(self) -> str:
        repr_str = super(FuzzyMaskGenerator, self).__repr__()
        repr_str += f"fuzzy_prob: {self.fuzzy_prob:.03f}\n"
        return repr_str

    def sample_mask_per_frame(self):
        """Produce a float-valued mask, where the revealed patch values are
        uniformly sampled in [0, 1] with p = self.fuzzy_prob and are
        0.0 with p = (1 - self.fuzzy_prob)
        """
        fuzzy_revealed = torch.rand([self.num_visible]) < self.fuzzy_prob
        fuzzy_revealed = fuzzy_revealed.float()
        fuzzy_revealed *= torch.rand([self.num_visible])
        mask = torch.cat(
            [fuzzy_revealed, torch.ones([self.num_masked], dtype=fuzzy_revealed.dtype)],
            dim=0,
        ).float()

        # get shuffled indices and shuffle the mask
        inds = torch.randperm(mask.shape[0]).long()
        mask = mask[inds]

        return mask


class HeterogeneousFrameMasker:
    """Produce masks of different ratios for different groups of frames.

    Example usage:
        frame_groups = [
            FrameGroup(mask_fraction=0, num_frames=1),
            FrameGroup(mask_fraction=0.99, num_frames=1),
        ]

        masker = HeterogeneousFrameMasker(
            height=height,
            width=width,
            frame_groups=frame_groups
        )

        masks = masker() # implicit batch_size = 1
        masks = masker(torch.rand(5, 32)) # batch_size = 5 ,  inferred from input

    The randomize_frame_groups flag determines whether the per-frame group masks
    are permuted every time a mask is generated. For example, setting this
    argument to True with the two_frame grouping setting would mean that the
    heavily masked frame is the first frame with p=0.5 and the second frame with p=0.5.
    """

    def __init__(
        self,
        height: int,
        width: int,
        frame_groups: list[FrameGroup],
        randomize_frame_groups: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.generators = [
            MaskGenerator(
                (group.num_frames, height, width),
                mask_fraction=group.mask_fraction,
                seed=seed,
            )
            for group in frame_groups
        ]

        self.randomize_frame_groups = randomize_frame_groups

    def __repr__(self) -> str:
        repr_str = ""
        for i, gen in enumerate(self.generators):
            repr_i = gen.__repr__()
            repr_str += f"Frame group {i}\n{repr_i}\n"
        return repr_str

    def set_num_visible(self, num_visible: int, frame: int = -1) -> None:
        frame = frame % len(self.generators)
        self.generators[frame].set_num_visible(num_visible)

    def set_mask_fraction(self, mask_fraction: float, frame: int = -1) -> None:
        frame = frame % len(self.generators)
        self.generators[frame].set_mask_fraction(mask_fraction)

    def __call__(self, x: Tensor | None = None, batch_size: int | None = None):
        mask_groups = [[generator(x=x, batch_size=batch_size)] for generator in self.generators]
        all_masks = list(itertools.chain.from_iterable(mask_groups))
        if self.randomize_frame_groups:
            inds = torch.randperm(len(all_masks))
            permuted_masks = torch.stack(all_masks, dim=-2)[..., inds, :]
            return permuted_masks.reshape(*permuted_masks.shape[:-2], -1)
        return torch.cat(all_masks, dim=-1)


class FuzzyHeterogeneousFrameMasker(HeterogeneousFrameMasker):
    def __init__(
        self,
        height: int,
        width: int,
        frame_groups: list[FrameGroup],
        fuzzy_probs: list[float] | float | None = None,
        randomize_frame_groups: bool = False,
        seed: int | None = None,
    ) -> None:
        if fuzzy_probs is None:
            fuzzy_probs = [0.0] * len(frame_groups)
        elif isinstance(fuzzy_probs, float):
            fuzzy_probs = [fuzzy_probs] * len(frame_groups)
        else:
            assert hasattr(fuzzy_probs, "__len__")
            assert len(fuzzy_probs) == len(frame_groups)

        self.generators = [
            FuzzyMaskGenerator(
                (group.num_frames, height, width),
                mask_fraction=group.mask_fraction,
                seed=seed,
                fuzzy_prob=fuzzy_probs[idx],
            )
            for idx, group in enumerate(frame_groups)
        ]

        self.randomize_frame_groups = randomize_frame_groups


class SoftBackgroundGenerator(MaskGenerator):
    """
    All visible patches have values in [0, reveal_max] and all masked patches have
    values in [mask_min, 1.0]
    """

    def __init__(self, *args, reveal_max: float = 0.5, mask_min: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.reveal_max = reveal_max
        self.mask_min = mask_min

    def sample_mask_per_frame(self):
        # create revealed patches with value [0, reveal_max]
        fuzzy_revealed = self.reveal_max * torch.rand(self.num_visible)

        # create background patches, each of which is in [mask_min, 1.0]
        background_deltas = torch.rand(self.num_masked) * (1 - self.mask_min)
        fuzzy_background = torch.ones(self.num_masked) - background_deltas

        mask = torch.cat(
            [fuzzy_revealed, fuzzy_background],
            dim=0,
        ).float()

        # get shuffled indices and shuffle the mask
        inds = torch.randperm(mask.shape[0]).long()
        mask = mask[inds]

        return mask


class SoftBackgroundFrameMasker(HeterogeneousFrameMasker):
    def __init__(
        self,
        height: int,
        width: int,
        frame_groups: list[FrameGroup],
        reveal_maxes: list[float] | None = None,
        mask_mins: list[float] | None = None,
        seed: int | None = None,
    ) -> None:
        reveal_maxes = reveal_maxes or [0, 0.5]
        mask_mins = mask_mins or [1.0, 0.9]
        self.generators = [
            SoftBackgroundGenerator(
                (group.num_frames, height, width),
                mask_fraction=group.mask_fraction,
                seed=seed,
                reveal_max=reveal_maxes[idx],
                mask_min=mask_mins[idx],
            )
            for idx, group in enumerate(frame_groups)
        ]


# short-cuts for common frame group settings
def two_frame(mask_fraction: float = 0.99) -> list[FrameGroup]:
    """Short-cut for not masking first frame, but masking the second frame by some
    amount."""
    return [
        FrameGroup(mask_fraction=0, num_frames=1),
        FrameGroup(mask_fraction=mask_fraction, num_frames=1),
    ]
