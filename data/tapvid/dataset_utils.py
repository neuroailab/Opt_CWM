import io
from collections import defaultdict

import numpy as np
from PIL import Image
from torch.utils.data.dataloader import default_collate


def decode_byte_array_imgs(img):
    """Decodes images from Kinetics pkl format."""
    byteio = io.BytesIO(img)
    img = Image.open(byteio)
    return np.array(img)


def collate_by_shape(batch):
    """Custom collate_fn to group videos by shape."""
    size_groups = defaultdict(list)

    for data in batch:
        h, w = data["videos"].size()[-2:]
        size_groups[(h, w)].append(data)

    batched_data = {}
    for i, group in enumerate(size_groups):
        data = default_collate(size_groups[group])
        data = {f"{i}_{k}": v for k, v in data.items()}
        batched_data.update(data)

    batched_data["num_video_groups"] = len(size_groups)

    return batched_data
