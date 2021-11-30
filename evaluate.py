"""
Main evaluation script for the pre-trained backbone models, a supervised task is chosen to perfect
the learnt representation and the performance on the validation set is evaluated.
"""

from dataclasses import dataclass
import os
from os import path
from pathlib import Path
import json
from typing import Union
import sys
import json

import torch
from torch import nn, optim
from torchvision import models, datasets
import torchvision.transforms as T

import geossl.parser as geoparser
from geossl.utils import is_port_in_use, make_serializable, OnlineKappa
from geossl.datasets import get_dataset_spec, create_dataset, MaskedDataset
from geossl.backbones import ResNetBackbone


@geoparser.dataparser
class EvaluateArgs:
    "Linear of finetuning evaluation of the models"

    data_dir: Path = geoparser.Field(positional=True)
    pretrained: Path = geoparser.Field(positional=True)

    weights: str = geoparser.Field(default="freeze", choices=("finetune", "freeze"))
    backbone_arch: str = geoparser.Field(
        default="resnet18", choices=("resnet18", "resnet50")
    )
    workers: int = int(os.getenv("SLURM_CPUS_PER_TASK", str(os.cpu_count() // 4)))
    n_gpus: int = 1
    n_epochs: int = 100
    batch_size: int = 256
    lr_backbone: float = 0.005
    lr_classifier: float = 0.3
    no_small_conv: bool = geoparser.Field(action="store_true")
    weight_decay: float = 1e-6
    training_ratio: float = 1.0
    checkpoint_dir: Path = Path("./checkpoint/lincls/")


@dataclass()
class TrainResult:
    top1: float = 0.0
    top5: float = 0.0
    kappa: float = 0.0


def main_worker(device, args: EvaluateArgs):
    torch.manual_seed(42)
    is_distributed = hasattr(args, "rank")

    if is_distributed:
        args.rank += device
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.n_gpus,
            rank=args.rank,
        )

    print(
        f"training with mode {args.weights} and {args.training_ratio * 100}% of samples"
    )

    if not is_distributed or args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if device != "cpu":
        torch.cuda.set_device(device)

    if "resisc" in str(args.data_dir):
        dataset_id = "resisc"
    elif "eurosat_rgb" in str(args.data_dir):
        dataset_id = "eurosat_rgb"
    elif "eurosat" in str(args.data_dir):
        dataset_id = "eurosat"
    else:
        raise NotImplementedError()
    dataset_spec = get_dataset_spec(dataset_id)

    # FIXME:
    # path.basename(path.split(args.data_dir)[0]))

    model = ResNetBackbone(
        resnet=args.backbone_arch,
        num_classes=dataset_spec.num_classes,
        with_classifier=True,
        small_conv=dataset_spec.crop_size < 100 and not args.no_small_conv,
    ).to(device)

    # state_dict = {
    #     key.replace("model.", ""): value
    #     for key, value in torch.load(args.pretrained, map_location="cpu").items()
    # }
    state_dict = torch.load(args.pretrained, map_location="cpu")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert (
        missing_keys == ["model.fc.weight", "model.fc.bias"] and unexpected_keys == []
    )

    model.model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.model.fc.bias.data.zero_()

    if args.weights == "freeze":
        model.requires_grad_(False)
        model.model.fc.requires_grad_(True)
    classifier_parameters, model_parameters = [], []
    for name, param in model.named_parameters():
        if name in {"model.fc.weight", "model.fc.bias"}:
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    criterion = nn.CrossEntropyLoss().to(device)

    param_groups = [dict(params=classifier_parameters, lr=args.lr_classifier)]
    if args.weights == "finetune":
        param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

    normalize = T.Normalize(mean=dataset_spec.mean, std=dataset_spec.std)

    train_dataset = create_dataset(
        args.data_dir,
        train=True,
        transform=T.Compose(
            [
                T.RandomResizedCrop(dataset_spec.crop_size),
                T.RandomHorizontalFlip(),
                *([T.ToTensor()] if dataset_id == "eurosat_rgb" else []),
                normalize,
            ]
        ),
    )
    real_len = len(train_dataset)
    train_dataset = MaskedDataset(train_dataset, ratio=args.training_ratio)
    assert len(train_dataset) == int(args.training_ratio * real_len)
    val_dataset = create_dataset(
        args.data_dir,
        train=False,
        transform=T.Compose(
            [
                T.Resize(dataset_spec.size),
                T.CenterCrop(dataset_spec.crop_size),
                *([T.ToTensor()] if dataset_id == "eurosat_rgb" else []),
                normalize,
            ]
        ),
    )

    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if is_distributed
        else None
    )
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device != "cpu",
        persistent_workers=False,
    )
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, shuffle=not is_distributed, 
        generator=g, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)
    best_acc = TrainResult()

    for epoch in range(args.n_epochs):
        if args.weights == "finetune":
            model.train()
        elif args.weights == "freeze":
            model.eval()

        epoch_loss = 0

        if is_distributed:
            train_sampler.set_epoch(epoch)
        for step, (images, target) in enumerate(
            train_loader, start=epoch * len(train_loader)
        ):
            output = model(images.to(device))
            loss = criterion(output, target.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        if not is_distributed or args.rank == 0:
            top1 = AverageMeter("Acc@1")
            top5 = AverageMeter("Acc@5")
            kappa = OnlineKappa(45)
            with torch.no_grad():
                for images, target in val_loader:
                    output = model(images.to(device))
                    target = target.to(device)
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
                    kappa.update(output.argmax(dim=-1), target)
            best_acc.top1 = max(best_acc.top1, top1.avg)
            best_acc.top5 = max(best_acc.top5, top5.avg)
            kappa_val = kappa.value()
            best_acc.kappa = max(best_acc.kappa, kappa_val)
            stats = dict(
                epoch=epoch,
                epoch_loss=epoch_loss,
                acc1=top1.avg,
                acc5=top5.avg,
                kappa=kappa_val,
                best_acc1=best_acc.top1,
                best_acc5=best_acc.top5,
                best_kappa=best_acc.kappa,
            )
            print(json.dumps(stats))
            sys.stdout.flush()

        scheduler.step()
        if not is_distributed or args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                best_acc=best_acc,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
            )
            torch.save(state, args.checkpoint_dir / "checkpoint.pth")
            with open(
                args.checkpoint_dir
                / f"results_{args.weights}_{args.training_ratio}.json",
                "w",
            ) as f:
                json.dump(make_serializable(best_acc.__dict__), f, indent=4)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    args = geoparser.from_args(EvaluateArgs)
    if args.n_gpus > 1:
        args.rank = 0
        port = 58472
        while is_port_in_use(port):
            port += 1
        args.dist_url = f"tcp://localhost:{port}"
        torch.multiprocessing.spawn(main_worker, (args,), args.n_gpus)
    else:
        device = "cuda:0" if args.n_gpus == 1 and torch.cuda.is_available() else "cpu"
        main_worker(device, args)


if __name__ == "__main__":
    main()
