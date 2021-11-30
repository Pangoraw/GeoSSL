import socket
from typing import Dict, Tuple

import torch
from torch import Tensor


def is_port_in_use(port):
    "Returns wether or not the current port is in use"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def make_serializable(obj):
    "Makes sures no value in the object are not json serializable"
    return {
        str(k): str(v) if not isinstance(v, (int, float)) else v for k, v in obj.items()
    }


class OnlineKappa:
    """
    Computes an online version of the Cohen's Kappa Coefficient.
    """

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.n_observations: int = 0
        self.n_agreed: int = 0
        self.data: Dict[int, Tuple[int, int]] = {
            cls: (0, 0) for cls in range(n_classes)
        }

    def update(self, y1: Tensor, y2: Tensor) -> int:
        assert y1.shape == y2.shape
        assert torch.all(y1 >= 0) and torch.all(y1 < self.n_classes) 
        assert torch.all(y2 >= 0) and torch.all(y2 < self.n_classes) 

        self.n_observations += y1.numel()
        for cls in range(self.n_classes):
            c1, c2 = self.data[cls]
            self.data[cls] = (
                int(c1 + (y1 == cls).sum().item()),
                int(c2 + (y2 == cls).sum().item()),
            )
        self.n_agreed += int((y1 == y2).sum())

        return self.n_observations

    def value(self) -> float:
        po = self.n_agreed / self.n_observations
        pe = sum([a * b for a, b in self.data.values()]) / (self.n_observations ** 2)

        return (po - pe) / (1 - pe)
