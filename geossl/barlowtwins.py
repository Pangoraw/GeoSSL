from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .base_method import BaseMethod
from .backbones import ResNetBackbone


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(BaseMethod):
    """Implementation of the BarlowTwins[1].

    [1]J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny, “Barlow Twins: Self-Supervised Learning via Redundancy Reduction,”
       arXiv:2103.03230 [cs, q-bio], Jun. 2021. Available: http://arxiv.org/abs/2103.03230
    """

    def __init__(
        self,
        backbone: ResNetBackbone,
        lambd: float,
        batch_size: int,
        h_dim: Optional[int] = None,
        num_proj_layers: int = 3,
    ):
        super(BarlowTwins, self).__init__()

        self.backbone = backbone

        z_dim = self.backbone.out_dim
        if h_dim is None:
            h_dim = 4 * z_dim

        sizes = [z_dim] + [h_dim for _ in range(num_proj_layers)]
        layers = []
        for i in range(len(sizes) - 2):
            layers += [
                nn.Linear(sizes[i], sizes[i + 1], bias=False),
                nn.BatchNorm1d(sizes[i + 1]),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projection_head = nn.Sequential(*layers)

        self.lambd = lambd
        self.batch_size = batch_size
        self.bn = nn.BatchNorm1d(h_dim, affine=False)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        z1 = self.projection_head(self.backbone(x1))
        z2 = self.projection_head(self.backbone(x2))

        c = self.bn(z1).T @ self.bn(z2)  # Compute the cross correlation matrix C
        c.div_(self.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.lambd * off_diag

        if torch.isnan(loss).item():
            raise Exception("loss is NaN")

        return loss
