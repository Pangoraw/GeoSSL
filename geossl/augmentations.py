from dataclasses import dataclass
from functools import partial
import random
from typing import Tuple, Union

from PIL import ImageFilter, Image
import torch
from torch import Tensor
import torchvision.transforms.functional as F
import torchvision.transforms as T


def odd(x: Union[int, float]) -> int:
    "Returns the closest odd integer"
    x = int(x)
    if x % 2 == 0:
        return x + 1
    return x


def random_gaussian(x: Tensor, kernel_size: int) -> Tensor:
    "Gaussian blur with a random sigma"
    sigma = (.1, 2.)[torch.randint(low=0, high=1, size=(1,)).item()]
    return F.gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)


@dataclass
class AugmentationSpecs:
    """
    A class used to control which augmentations are used in the
    augmentation pipeline.
    """
    gray_scale: bool = True
    color_jitter: bool = True
    flip: bool = True
    crop: bool = True

    @staticmethod
    def from_str(s: str):
        "Builds an AugmentationSpecs from a string with pattern '0110'"
        assert len(s) == 4
        return AugmentationSpecs(*(c == '1' for c in s))


class SimCLRAugmentations(object):
    """
    The set of augmentations from SimCLR
    """
    def __init__(
        self,
        size: int,
        s: float = 0.1,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        specs: AugmentationSpecs=AugmentationSpecs(),
        move_to_tensor: bool = False,
    ):
        augs = []

        print(f">> Building augmentations {specs}")

        if specs.crop:
            augs.append(T.RandomResizedCrop(size=size, scale=(.2, 1.)))

        if specs.flip:
            augs.append(T.RandomHorizontalFlip())
            augs.append(T.RandomVerticalFlip())

        if specs.color_jitter:
            augs.append(
                T.RandomApply([T.ColorJitter(0.4*s, 0.4*s, 0.4*s, 0.1*s)], p=0.8),
            )

        if specs.gray_scale:
            augs.append(T.RandomGrayscale(p=0.2))

        if move_to_tensor:
            augs.append(T.ToTensor())

        self.augment = T.Compose(augs + [
            # NOTE: No random Gaussian for eurosat
            # T.RandomApply(
            #     [partial(random_gaussian, kernel_size=odd(.1 * size))], p=0.5),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.augment(x), self.augment(x)


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class BarlowTwinsAugmentations:
    def __init__(self, size: int, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.augment = T.Compose([
            T.RandomResizedCrop(size, interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            # GaussianBlur(p=1.),
            T.Normalize(mean=mean,
                        std=std)
        ])
        self.augment_prime = T.Compose([
            T.RandomResizedCrop(size, interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            # GaussianBlur(p=0.1),
            # T.RandomApply([partial(F.solarize, threshold=128.)]),
            # T.ToTensor(),
            T.Normalize(mean=mean,
                        std=std),
        ])

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.augment(x), self.augment_prime(x)
