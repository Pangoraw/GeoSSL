from functools import partial
import random
from typing import Tuple

from PIL import ImageFilter, Image
import torch
from torch import Tensor
import torchvision.transforms.functional as F
import torchvision.transforms as T


def random_gaussian(x: Tensor, kernel_size: int) -> Tensor:
    "Gaussian blur with a random sigma"
    sigma = (.1, 2.)[torch.randint(low=0, high=1, size=(1,)).item()]
    return F.gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)


class SimCLRAugmentations(object):
    def __init__(self, size: int, s: float = 0.1):
        self.augment = T.Compose([
        T.RandomResizedCrop(size=size),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([partial(random_gaussian, kernel_size=int(.1 * size))], p=0.5),
        T.ToTensor(),
    ])

    def __call__(self, x: Tensor) -> Tuple[Tensor,Tensor]:
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


class BarlowTwinsAugmentations(object):
    def __init__(self, size: int):
        self.augment = T.Compose([
            T.RandomResizedCrop(size, interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
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
            GaussianBlur(p=0.1),
            T.RandomApply([partial(F.solarize, threshold=128.)]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x: Tensor) -> Tuple[Tensor,Tensor]:
        return self.augment(x), self.augment_prime(x)
