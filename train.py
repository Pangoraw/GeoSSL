import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from geossl import  SimCLR, ResNetBackbone
from geossl.augmentations import SimCLRAugmentations
from geossl.optimizer import LARS


def main_worker(gpu, args):
    torch.cuda.set_device(gpu)

    backbone = ResNetBackbone(args.backbone_arch)

    model = SimCLR(backbone, tau=1.).cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    opt = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    augment = SimCLRAugmentations(224)
    train_dataset = ImageFolder(args.train_dir, transform=T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        augment,
    ]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
    )

    def save_checkpoint():
        checkpoint_path = args.checkpoint_dir / "checkpoint.pth"
        state_dict = backbone.state_dict()
        torch.save(state_dict, checkpoint_path)
        print(f">> Saved checkpoint at {checkpoint_path}")

    print(f">> Training with {len(train_dataset)} images")

    for epoch in range(args.n_epochs):
        batch_loss = 0

        for (x1, x2), _ in train_loader:
            x1 = x1.cuda(gpu, non_blocking=True)
            x2 = x2.cuda(gpu, non_blocking=True)
            loss = model(x1, x2)
            batch_loss += loss.item()

            model.zero_grad()
            loss.backward()
            opt.step()

        if epoch % 2 == 0:
            print(f">> [Epoch {epoch}/{args.n_epochs}] loss = {batch_loss:0.4f}")

        if epoch % 10 == 0:
            save_checkpoint()

    save_checkpoint()


def main():
    parser = argparse.ArgumentParser("SSL trainer")
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--backbone_arch", default="resnet18", choices=["resnet18", "resnet50"])
    args = parser.parse_args()
    torch.multiprocessing.spawn(main_worker, (args,), args.n_gpus)


if __name__ == "__main__":
    main()
