import datetime
import json
import os
from pathlib import Path
from typing import Optional
import shutil

import torch

from geossl.utils import is_port_in_use, make_serializable
import geossl.parser as geoparser

from train import TrainingArgs, main_worker as train
from evaluate import EvaluateArgs, main_worker as evaluate


@geoparser.dataparser
class TrainEvalArgs:
    "A script that both trains and evaluates a model"

    data_dir: Path = geoparser.Field(positional=True)
    backbone_arch: str = geoparser.Field(
        default="resnet18", choices=["resnet18", "resnet50"])
    method: str = geoparser.Field(
        default="simclr", choices=["simclr", "barlow", "byol", "moco"])
    train_batch_size: int = 512
    train_optimizer: str = geoparser.Field(default="sgd", choices=["sgd", "lars"])
    train_learning_rate_weights: float = 0.05
    train_learning_rate_biases: float = 0.05
    train_cosine_schedule: bool = geoparser.Field(action="store_true")
    checkpoint_dir: Path = geoparser.Field(
        default=None,
        help="Override the checkpoint directory, useful to resume the process"
    )
    augmentation_specs: Optional[str] = None
    force_eval: bool = geoparser.Field(action="store_true", help="Force evaluation")
    n_epochs: int = 1_000
    no_small_conv: bool = geoparser.Field(action="store_true",
            help="Disable removing the maxpooling for small input sizes")


def run_dir(args: TrainEvalArgs) -> Path:
    "Creates a unique directory to save the run artifacts and return its path"
    name = args.data_dir.name
    date = datetime.datetime.now().strftime("%y-%m-%dT%H-%M-%S")
    identifier = os.getenv("SLURM_JOBID", date)
    p = f"./checkpoints/train-{args.method}-{identifier}-{name}"
    if args.augmentation_specs is not None:
        p += f"-{args.augmentation_specs}"
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_with(gpu, args: EvaluateArgs, **kwargs):
    "Overrides the evaluation args with kwargs"
    new_args = EvaluateArgs(**{**args.__dict__, **kwargs})
    return evaluate(gpu, new_args)


def main(args: TrainEvalArgs):
    if args.checkpoint_dir is None:
        args.checkpoint_dir = run_dir(args)

    training_args = TrainingArgs(
        train_dir=args.data_dir / "train" if "eurosat_torchgeo" in str(args.data_dir) else args.data_dir,
        n_epochs=args.n_epochs,
        batch_size=args.train_batch_size,
        optimizer=args.train_optimizer,
        backbone_arch=args.backbone_arch,
        method=args.method,
        checkpoint_dir=args.checkpoint_dir,
        learning_rate_weights=args.train_learning_rate_weights,
        learning_rate_biases=args.train_learning_rate_biases,
        cosine=args.train_cosine_schedule,
        temperature=0.07,
        augmentation_specs=args.augmentation_specs,
        no_small_conv=args.no_small_conv,
    )

    checkpoint_path = args.checkpoint_dir / "checkpoint.pth"
    evaluate_args = EvaluateArgs(
        data_dir=args.data_dir,
        pretrained=checkpoint_path,
        backbone_arch="resnet18",
        lr_classifier=0.3,
        lr_backbone=0.25,
        batch_size=64,
        no_small_conv=args.no_small_conv,
    )

    args_file = args.checkpoint_dir / "args.json"
    if not args_file.exists():
        all_args = { # dump config to checkpoint folder
            "global_args": make_serializable(args.__dict__),
            "training_args": make_serializable(training_args.__dict__),
            "evaluate_args": make_serializable(evaluate_args.__dict__),
        }
        with open(args_file, "w") as f:
            json.dump(all_args, f, indent=4)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if not checkpoint_path.exists():
        print(f"Training in folder {args.checkpoint_dir}...")
        train(device, training_args)
        print("Training done; Evaluating...")
    else:
        print(f"Checkpoint file ({checkpoint_path}) already exists, skipping training...")

    assert checkpoint_path.exists()

    # Do the three evaluation benchmarks (linear 100%; finetune 10%; finetune 1%)
    checkpoint_dir = args.checkpoint_dir / "freeze-1.0"
    if args.force_eval and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    if not checkpoint_dir.exists():
        evaluate_with(
            device,
            evaluate_args,
            weights="freeze",
            training_ratio=1.0,
            checkpoint_dir=checkpoint_dir,
            lr_classifier=0.05,
            lr_backbone=0.05,
        )

    checkpoint_dir = args.checkpoint_dir / "finetune-0.1"
    if args.force_eval and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    if not checkpoint_dir.exists():
        evaluate_with(
            device,
            evaluate_args,
            weights="finetune",
            training_ratio=0.1,
            checkpoint_dir=checkpoint_dir,
        )

    checkpoint_dir = args.checkpoint_dir / "finetune-0.01"
    if args.force_eval and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    if not checkpoint_dir.exists():
        evaluate_with(
            device,
            evaluate_args,
            weights="finetune",
            training_ratio=0.01,
            checkpoint_dir=checkpoint_dir,
        )

    print("Evaluating done")


if __name__ == "__main__":
    args = geoparser.from_args(TrainEvalArgs)
    main(args)
