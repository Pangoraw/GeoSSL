from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchgeo.datasets import EuroSAT as TorchgeoEuroSAT, RESISC45

from .eurosat import EuroSAT
from .resisc import Resisc45


def eurosat_to_rgb(x: Tensor) -> Tensor:
    x = x[[3, 2, 1], :, :].float()
    x.sub_(x.min())
    return x.div_(x.max())


class EuroSATRGB(Dataset):
    """
    Returns a RGB version of the EuroSAT dataset with the same split
    as in torchgeo.
    """

    def __init__(self, root: Union[str, Path], transform, **kwargs):
        super(EuroSATRGB, self).__init__()
        self.transform = transform
        self.eurosat = TorchgeoEuroSAT(root, **kwargs)

    def _process_item(self, x) -> Tuple[Tensor, int]:
        img = x["image"].float()
        label = x["label"]
        return self.transform(eurosat_to_rgb(img)), label

    def __len__(self):
        return len(self.eurosat)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        return self._process_item(self.eurosat[index])


class RESISC45_Wrapper(Dataset):
    "A NWPU-RESISC45 wrapper for a more torchvision like API."

    def __init__(self, root, transform, **kwargs):
        super(RESISC45_Wrapper, self).__init__()
        self.resisc = RESISC45(root, **kwargs)
        self.transform = transform

    def _process_item(self, x) -> Tuple[Tensor, int]:
        img = x["image"] / 255
        label = x["label"]
        return self.transform(img), label

    def __len__(self):
        return len(self.resisc)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        return self._process_item(self.resisc[index])


class MaskedDataset(Dataset):
    "A dataset that returns only ratio% of samples"

    def __init__(self, dataset: Dataset, ratio: float = 1.0):
        super(MaskedDataset, self).__init__()
        assert 0.0 < ratio <= 1.0

        self.ratio = ratio
        self.dataset = dataset

        n_samples = int(self.ratio * len(self.dataset))
        self.range = torch.arange(len(self.dataset))[torch.randperm(len(self.dataset))][
            :n_samples
        ]

    def __len__(self):
        return self.range.size(0)

    def __getitem__(self, idx):
        new_idx = self.range[idx].item()
        return self.dataset[new_idx]


@dataclass()
class DatasetSpec:
    num_classes: int
    size: int
    crop_size: int

    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


def create_dataset(root, **kwargs) -> Dataset:
    if "eurosat_rgb" in str(root):
        kwargs["split"] = "train" if kwargs.pop("train", True) else "val"
        return EuroSAT(root, download=True, **kwargs)
    if "eurosat" in str(root):
        kwargs["split"] = "train" if kwargs.pop("train", True) else "val"
        return EuroSATRGB(root, download=True, **kwargs)
    if "resisc_rgb" in str(root):
        kwargs["split"] = "train" if kwargs.pop("train", True) else "val"
        return Resisc45(root, download=True, **kwargs)
    if "resisc" in str(root):
        kwargs["split"] = "train" if kwargs.pop("train", True) else "val"
        return RESISC45_Wrapper(root, download=True, **kwargs)
    return ImageFolder(root, **kwargs)


def get_dataset_spec(dataset: str) -> DatasetSpec:
    "Returns the dataset spec for the given path"

    if "eurosat_rgb" in str(dataset):
        return DatasetSpec(
            num_classes=10,
            size=224, # TODO: put back to 64 once the transfer test is done
            crop_size=224, # also
            mean=(0.3448, 0.3807, 0.4082),
            std=(0.2037, 0.1369, 0.1151),
        )
    if "eurosat" in str(dataset):
        return DatasetSpec(
            num_classes=10,
            size=64,
            crop_size=64,
            mean=(0.2434, 0.3723, 0.4507),
            std=(0.2059, 0.1866, 0.2438),
        )
    if "resisc_rgb" in str(dataset):
        return DatasetSpec(
            num_classes=45,
            size=256,
            crop_size=224,
            mean=(0.3682, 0.3808, 0.3434),
            std=(0.2034, 0.1852, 0.1846),
        )
    if "resisc" in str(dataset):
        return DatasetSpec(
            num_classes=45,
            size=256,
            crop_size=224,
            mean=(0.3640, 0.3780, 0.3375),
            std=(0.2089, 0.1858, 0.1830),
        )
    if "cifar" in str(dataset):
        return DatasetSpec(
            num_classes=10,
            size=32,
            crop_size=32,
            mean=(0.4915, 0.4822, 0.4466),
            std=(0.2470, 0.2435, 0.2616),
        )
    raise NotImplementedError(dataset)
