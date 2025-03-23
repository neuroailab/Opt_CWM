import torch


class MultiMaskingGenerator:
    def __init__(self, input_size, mask_ratio, masking_iters):
        self.frames, self.height, self.width = input_size
        self.mask_ratio = mask_ratio
        self.masking_iters = masking_iters

        self.num_patches_per_frame = self.height * self.width
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)

    def __call__(self, batch_size=None):
        bs = 1 if batch_size is None else batch_size
        viz_frame = torch.zeros((bs, self.masking_iters, self.num_patches_per_frame))
        masked_frame = torch.clone(viz_frame)
        masked_frame[..., : self.num_masks_per_frame] = 1

        for b in range(bs):
            for m in range(self.masking_iters):
                masked_frame[b, m] = masked_frame[b, m, torch.randperm(self.num_patches_per_frame)]

        mask = torch.cat((viz_frame, masked_frame), -1)
        return mask if batch_size is not None else mask[0]
