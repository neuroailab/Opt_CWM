# Train two stream, add two stream eval code
import datetime
import os
import sys

import torch
import torch.distributed as dist
from tqdm import tqdm

# import wandb
from data.kinetics import dataset as kinetics_dataset
from models import builder
from utils import dist_logging, optim_utils, options, utils

logger = dist_logging.get_logger(__name__)


def _setup(meta_args, data_args, model_args, optim_args):
    torch.backends.cuda.enable_flash_sdp(True)

    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=meta_args.nccl_timeout))
    rank = dist.get_rank()

    # if rank == 0:
    #     wandb.init()

    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = local_rank
    torch.cuda.set_device(device_id)

    rnd_seed = meta_args.seed
    # random.seed(rnd_seed)
    # np.random.seed(rnd_seed)
    # torch.random.manual_seed(rnd_seed)
    utils.set_seed(meta_args.seed)

    model = builder.get_opt_cwm(model_args, device_id=device_id)
    model.load_pretrained(force=model_args.build.force)
    # model, optim_ckpt = checkpoint.load_opt_cwm_checkpoint(
    #     model, model_args.build.opt_cwm_ckpt, model_args.build.base_cwm_ckpt, model_args.build.load_optim
    # )

    model.flow_predictor.cwm_model.requires_grad_(False)
    logger.info("Freezing base_cwm")

    model = model.cuda().to(device_id)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id])

    optimizer = optim_utils.create_optimizer(optim_args, model.module)

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    total_batch_size = data_args.batch_size * dist.get_world_size()  # assuming no grad accumulation

    lr_schedule, wd_schedule = None, None

    lr = optim_args.lr * total_batch_size / 256
    min_lr = optim_args.min_lr * total_batch_size / 256

    if optim_args.scheduler == "cosine":
        lr_schedule = optim_utils.cosine_scheduler(
            base_value=lr,
            final_value=min_lr,
            epochs=optim_args.epochs,
            niter_per_ep=data_args.samples_per_epoch // data_args.batch_size,
            warmup_epochs=optim_args.warmup_epochs,
            warmup_steps=optim_args.warmup_steps,
        )

        wd_schedule = optim_utils.cosine_scheduler(
            base_value=optim_args.weight_decay,
            final_value=optim_args.weight_decay_end,
            epochs=optim_args.epochs,
            niter_per_ep=data_args.samples_per_epoch // data_args.batch_size,
        )
    itr = 0

    loader = kinetics_dataset.get_kinetics_loader(
        augmentation_type=data_args.kinetics.augmentation_type,
        augmentation_scales=data_args.kinetics.augmentation_scales,
        crop_size=data_args.kinetics.crop_size,
        data_path=data_args.kinetics.path_to_txt,
        frame_delta_ms=data_args.kinetics.frame_delta_ms,
        samples_per_epoch=data_args.samples_per_epoch,
        batch_size=data_args.batch_size,
        num_workers=data_args.num_workers,
    )

    return model, loader, lr_schedule, wd_schedule, itr, optimizer, scaler


def run_epoch(model, optimizer, scaler, loader, itr, lr_schedule, wd_schedule):
    model.train()

    criterion = torch.nn.MSELoss()

    for idx, video in enumerate(tqdm(loader, desc=f"train run (itr {itr})", disable=dist.get_rank())):
        optim_utils.set_lr_and_wd(optimizer, lr_schedule, wd_schedule, itr)
        video = video.cuda()

        with torch.cuda.amp.autocast(enabled=True):
            frame1_recon = model(video)

            patch_size = utils.extract_attributes_from_maybe_ddp(model, "patch_size")

            videos_unnorm_patchified = utils.video_to_patches(
                utils.imagenet_unnormalize(video[:, :, 1:2]), patch_t=1, patch_h=patch_size, patch_w=patch_size
            )

            mean_loss = criterion(frame1_recon, videos_unnorm_patchified)  # single scalar

        scaler.scale(mean_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        if idx % 25 == 0:
            mean_loss = mean_loss.detach()
            mean_loss = utils.gather_tensor(mean_loss.unsqueeze(0))  # tensor of size WORLD_SIZE

            # if dist.get_rank() == 0:
            #     mean_loss = torch.mean(mean_loss)

            #     wandb.log({"iter/batch_loss": mean_loss.item(), "iter": itr})
            #     wandb.log({"iter/batch_loss_log": torch.log(mean_loss).item()})
            #     wandb.log({"iter/lr": lr_schedule[itr], "iter": itr})

        itr += 1

    return itr


if __name__ == "__main__":
    opt_cmd = options.parse_arguments(sys.argv[1:])
    train_cfg = options.set(opt_cmd=opt_cmd, verbose=False)

    data_args = train_cfg.data_args
    meta_args = train_cfg.meta_args
    model_args = train_cfg.model_args
    optim_args = train_cfg.optim_args

    model, train_loader, lr_schedule, wd_schedule, itr, optimizer, scaler = _setup(
        meta_args, data_args, model_args, optim_args
    )

    start_epoch = itr // (data_args.samples_per_epoch // data_args.batch_size)

    for epoch in range(start_epoch, optim_args.epochs):
        itr = run_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loader=train_loader,
            itr=itr,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
        )
