import torch
from torch import nn, Tensor
import torch.nn.functional as F


def off_diagonal(x: Tensor) -> Tensor:
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    """Implementation of the BarlowTwins[1].

    [1]J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny, “Barlow Twins: Self-Supervised Learning via Redundancy Reduction,”
       arXiv:2103.03230 [cs, q-bio], Jun. 2021, Accessed: Nov. 03, 2021. [Online]. Available: http://arxiv.org/abs/2103.03230
    """

    def __init__(self, backbone: nn.Module, lambd: float):
        super(BarlowTwins, self).__init__()
        self.backbone = backbone
        self.lambd = lambd

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        N, D = z1.shape

        c = (F.normalize(z1, dim=0).T @ F.normalize(z2, dim=0)) / N
        c_diff = (c - torch.eye(D, device=c.device)).pow(2)
        off_diagonal(c_diff).mul_(self.lambd)

        return c_diff.sum()
