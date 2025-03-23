import os
import sys
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from torchvision import transforms

from models import builder
from utils import constants, options, utils

DEVICE = "cuda:0"
DEMO_DIR = "demo"


opt_cmd = options.parse_arguments(sys.argv[1:])
eval_cfg = options.set(opt_cmd=opt_cmd, verbose=False)

model_args = eval_cfg.model_args

torch.cuda.set_device(DEVICE)

model = builder.get_flow_predictor(model_args)
model.load_pretrained(model_args.build.highres, model_args.build.force)
model.to(DEVICE)


def get_video():

    image_files = glob(f"{DEMO_DIR}/penguin_*.png")
    image_files = sorted(image_files)

    point_file = glob(f"{DEMO_DIR}/penguin_*.npy")[0]
    points = np.load(point_file)

    points = torch.LongTensor(points).to(DEVICE)

    frame0, frame1 = Image.open(image_files[0]), Image.open(image_files[-1])

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(constants.IMAGENET_DEFAULT_MEAN, constants.IMAGENET_DEFAULT_STD),
        ]
    )

    frames = torch.stack([transform(x) for x in [frame0, frame1]], dim=1).unsqueeze(0)
    frames = frames.repeat_interleave(points.size(0), 0).to(DEVICE)

    return frames, points


@torch.no_grad()
def run():
    video, points = get_video()

    with torch.cuda.amp.autocast(enabled=True):
        out = model(video, points)

    pred = out["pred_pixel_loc"].cpu().numpy()
    points = points.cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    video = utils.imagenet_unnormalize(video)

    f0, f1 = video[0].unbind(1)
    f0 = transforms.ToPILImage()(f0)
    f1 = transforms.ToPILImage()(f1)

    colors = colormaps.get_cmap("rainbow")(np.linspace(0, 1, points.shape[0]))

    axs[0].imshow(f0)
    axs[1].imshow(f1)

    axs[0].scatter(points[:, 1], points[:, 0], c=colors, s=10)
    axs[1].scatter(pred[:, 1], pred[:, 0], c=colors, s=10)

    for i in range(points.shape[0]):
        axs[0].arrow(
            points[i, 1],
            points[i, 0],
            pred[i, 1] - points[i, 1],
            pred[i, 0] - points[i, 0],
            fc=colors[i],
            ec=colors[i],
            head_width=10,
            head_length=10,
            alpha=0.6,
        )

    fig.tight_layout()
    fig.savefig("demo_result.png")

    print("Saved result to demo_result.png")

    plt.close(fig)


if __name__ == "__main__":
    run()
