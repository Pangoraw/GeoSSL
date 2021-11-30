"""
Main pre-training script for the backbone models (self-supervised training).
"""

import math
import json
import os
from pathlib import Path
import sys
import time
from typing import Optional

import torch
from torch import optim
import torchvision.transforms as T

from geossl import BYOL, MoCo, SimCLR, BarlowTwins, ResNetBackbone
from geossl.augmentations import SimCLRAugmentations, AugmentationSpecs
from geossl.augmentations import (
    SimCLRAugmentations,
    AugmentationSpecs,
)
import geossl.parser as geoparser
from geossl.utils import is_port_in_use
from geossl.datasets import get_dataset_spec, create_dataset
from geossl.optimizer import LARS


@geoparser.dataparser
class TrainingArgs:
    "A SSL Trainer"
    train_dir: Path = geoparser.Field(positional=True, help="The training dataset path")

    backbone_arch: str = geoparser.Field(
        default="resnet18", choices=["resnet18", "resnet50"]
    )
    method: str = geoparser.Field(
        default="simclr", choices=["simclr", "barlow", "byol", "moco"]
    )
    temperature: float = geoparser.Field(
        default=0.07, help="Temperature for the nt_xent loss [default=0.07]"
    )
    optimizer: str = geoparser.Field(default="sgd", choices=["sgd", "lars"])

    n_gpus: int = 1
    n_epochs: int = 10
    batch_size: int = 16
    checkpoint_dir: Path = Path("checkpoint/")
    weight_decay: float = 1e-4
    learning_rate_weights: float = 0.2
    learning_rate_biases: float = 0.0048
    cosine: bool = geoparser.Field(action="store_true")
    no_small_conv: bool = geoparser.Field(action="store_true")
    augmentation_specs: Optional[str] = None


def adjust_learning_rate(args, optimizer, loader, step):
    "From https://github.com/facebookresearch/barlowtwins"

    if args.cosine:
        max_steps = args.n_epochs * len(loader)
        warmup_steps = 10 * len(loader)
        base_lr = args.batch_size / 256
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            step -= warmup_steps
            max_steps -= warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)
        optimizer.param_groups[0]["lr"] = lr * args.learning_rate_weights
        optimizer.param_groups[1]["lr"] = lr * args.learning_rate_biases
    else:
        lr_decay_steps = [700, 800, 900]
        lr_decay_rate = 0.1

        n_epochs = step // len(loader)
        steps = sum(n_epochs > x for x in lr_decay_steps)

        lr = lr_decay_rate ** steps
        optimizer.param_groups[0]["lr"] = lr * args.learning_rate_weights
        optimizer.param_groups[1]["lr"] = lr * args.learning_rate_biases


def main_worker(device: torch.device, args: TrainingArgs):
    is_distributed = hasattr(args, "rank")
    torch.manual_seed(42)

    if is_distributed:
        assert device != "cpu", "Cannot use distributed with cpu"
        args.rank += device
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            rank=args.rank,
            world_size=args.n_gpus,
        )

    if device != "cpu":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    if "resisc" in str(args.train_dir):
        dataset_id = "resisc"
    elif "eurosat_rgb" in str(args.train_dir):
        dataset_id = "eurosat_rgb"
    elif "eurosat" in str(args.train_dir):
        dataset_id = "eurosat"
    else:
        raise NotImplementedError()

    dataset_spec = get_dataset_spec(dataset_id)
    img_size, crop_size = dataset_spec.size, dataset_spec.crop_size

    if args.augmentation_specs is not None:
        aug_specs = AugmentationSpecs.from_str(args.augmentation_specs)
    else:
        aug_specs = AugmentationSpecs()
    augment = SimCLRAugmentations(
        size=crop_size,
        mean=dataset_spec.mean,
        std=dataset_spec.std,
        specs=aug_specs,
        move_to_tensor=dataset_id == "eurosat_rgb",
    )

    img_transform = T.Compose([T.Resize(img_size), T.CenterCrop(crop_size), augment,])

    print("creating dataset " + str(args.train_dir))
    train_dataset = create_dataset(args.train_dir, train=True, transform=img_transform,)
    sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if is_distributed
        else None
    )
    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count() // 4))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size // args.n_gpus,
        sampler=sampler,
        shuffle=not is_distributed,
        num_workers=num_workers,
        pin_memory=device != "cpu",
        persistent_workers=False,
    )

    small_conv = img_size < 100 and not args.no_small_conv
    backbone = ResNetBackbone(args.backbone_arch, small_conv=small_conv)
    if args.method == "simclr":
        model = SimCLR(backbone, tau=args.temperature)
    elif args.method == "barlow":
        model = BarlowTwins(
            backbone,
            lambd=0.0051,
            batch_size=args.batch_size,
            h_dim=backbone.out_dim * (2 if small_conv else 4),
        )
    elif args.method == "byol":
        encoder_s = backbone
        encoder_t = ResNetBackbone(args.backbone_arch, small_conv=img_size < 100)
        base_momentum, final_momentum = 1.0, 0.99
        model = BYOL(
            encoder_s,
            encoder_t,
            base_momentum=base_momentum,
            final_momentum=final_momentum,
            n_steps=len(train_loader) * args.n_epochs,
        )
    elif args.method == "moco":
        encoder_s = backbone
        encoder_t = ResNetBackbone(args.backbone_arch, small_conv=img_size < 100)
        base_momentum, final_momentum = 1.0, 0.99
        model = MoCo(
            encoder_s,
            encoder_t,
            feat_dim=512,
            base_momentum=base_momentum,
            final_momentum=final_momentum,
            n_steps=len(train_loader) * args.n_epochs,
            queue_size=6 * args.batch_size,
        )
    else:
        raise NotImplementedError(args.method)

    model = model.to(device)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{"params": param_weights}, {"params": param_biases}]
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True
        )

    if args.optimizer == "sgd":
        opt = optim.SGD(parameters, 0, momentum=0.9, weight_decay=args.weight_decay)
    else:
        opt = LARS(
            parameters,
            lr=0,
            weight_decay=args.weight_decay,
            weight_decay_filter=True,
            lars_adaptation_filter=True,
        )
    model.train()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint():
        if is_distributed and args.rank != 0:
            return
        checkpoint_path = args.checkpoint_dir / "checkpoint.pth"
        state_dict = backbone.state_dict()
        torch.save(state_dict, checkpoint_path)
        print(f">> Saved checkpoint at {checkpoint_path}")

    if not is_distributed or args.rank == 0:
        stats_file = open(args.checkpoint_dir / "stats.txt", "w", buffering=1)
        start_time = time.time()

        print(
            f">> Training with {len(train_dataset)} images of size {dataset_spec.size}x{dataset_spec.size} "
            f"on {num_workers} workers"
        )

    for epoch in range(args.n_epochs):
        batch_loss = 0

        if sampler is not None and (not is_distributed or args.rank == 0):
            sampler.set_epoch(epoch)
        for step, ((x1, x2), _) in enumerate(
            train_loader, start=epoch * len(train_loader)
        ):
            x1 = x1.to(device)
            x2 = x2.to(device)

            if not is_distributed or args.rank == 0:
                adjust_learning_rate(args, opt, train_loader, step)

            loss = model(x1, x2)
            batch_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if not is_distributed or args.rank == 0:
                if is_distributed:
                    model.module.step(step)
                else:
                    model.step(step)

        if epoch % 2 == 0 and (not is_distributed or args.rank == 0):
            print(f">> [Epoch {epoch}/{args.n_epochs}] loss = {batch_loss:0.4f}")
            sys.stdout.flush()
            print(
                json.dumps(
                    dict(
                        loss=batch_loss,
                        lr=opt.param_groups[0]["lr"],
                        epoch=epoch,
                        time=int(time.time() - start_time),
                    )
                ),
                file=stats_file,
            )

        if epoch % 10 == 0 and (not is_distributed or args.rank == 0):
            save_checkpoint()

    if not is_distributed or args.rank == 0:
        save_checkpoint()


def main():
    args = geoparser.from_args(TrainingArgs)
    args.rank = 0
    port = 58472
    while is_port_in_use(port):
        port += 1
    args.dist_url = f"tcp://localhost:{port}"
    torch.multiprocessing.spawn(main_worker, (args,), args.n_gpus)


if __name__ == "__main__":
    main()
