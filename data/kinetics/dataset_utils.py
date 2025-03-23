from torchvision import transforms

from data.kinetics.transforms import *
from utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class DataAugmentationForVideoMAE(object):
    def __init__(self, augmentation_type, crop_size, augmentation_scales):

        transform_list = []

        self.scale = GroupScale(crop_size)
        transform_list.append(self.scale)

        if augmentation_type == "multiscale":
            self.train_augmentation = GroupMultiScaleCrop(crop_size, list(augmentation_scales))
        elif augmentation_type == "center":
            self.train_augmentation = GroupCenterCrop(crop_size)

        transform_list.extend([self.train_augmentation, Stack(roll=False), ToTorchFormatTensor(div=True)])

        # Normalize input images
        normalize = GroupNormalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        transform_list.append(normalize)

        self.transform = transforms.Compose(transform_list)

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr
