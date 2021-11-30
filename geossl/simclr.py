import warnings

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .backbones import ResNetBackbone
from .base_method import BaseMethod


LARGE_NUM = 1e9


def nt_xent(z: Tensor, perm: Tensor, tau: float) -> Tensor:
    r"""
    Pairwise normalized temperature scaled cross entropy loss (NT-Xent) loss
    Parameters
    ==========
    z: Tensor - size (2n, d) the concatenated projected features
    perm: Tensor - size (2n,) the positive pairs permutations in p (labels)
    tau: float - the temperature parameter

    Returns
    =======
    loss: Tensor - the NT-Xent loss
    """
    # 1. Compute the cosine similarity
    features = F.normalize(z, dim=1)
    sim = features @ features.T  # sim.shape == (2b, 2b)

    # 2. Set exp(sim[i,k] / tau) = 0. for i == k
    torch.diagonal(sim).sub_(LARGE_NUM)

    # 3. Apply temperature for other similarities
    sim /= tau

    # 4. Cross-entropy
    return F.cross_entropy(sim, perm)


class SimCLR(BaseMethod):
    """
    An implementation of SimCLR[1].

    [1]T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A Simple Framework for Contrastive Learning of Visual Representations,”
       arXiv:2002.05709 [cs, stat], Jun. 2020, Accessed: Nov. 02, 2021. [Online]. Available: http://arxiv.org/abs/2002.05709
    """

    def __init__(
        self,
        backbone: ResNetBackbone,
        tau: float,
        feat_dim: int = 128,
        loss: str = "nt_xent",
    ):
        super(SimCLR, self).__init__()
        self.backbone = backbone

        z_dim = self.backbone.out_dim
        self.projection_head = nn.Sequential(
            nn.Linear(z_dim, z_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim, feat_dim, bias=False),
        )
        self.tau = tau

    def forward(
        self, x1: Tensor, x2: Tensor
    ) -> Tensor:  # x1.shape == x2.shape == (b, c, h, w)
        """
        Returns the pairwise NT-Xent loss between x1 and x2
        Parameters
        ==========
            x1: Tensor - size (n, d) random augmentation of x
            x2: Tensor - size (n, d) random augmentation of x
        Returns
        =======
            loss: Tensor - the NT-Xent loss
        """
        b = x1.size(0)
        xp = torch.cat((x1, x2))  # xp.shape == (2b, c, h, w)

        # match each x1 with its corresponding x2
        # perm = torch.tensor([b, b+1, b+2,...b+b-1, 0, 1, 2,..., b-1])
        perm = torch.cat((torch.arange(b) + b, torch.arange(b)), dim=0).to(xp.device)

        h = self.backbone(xp)  # h.shape == (2b, C)
        z = self.projection_head(h)  # p.shape == (2b, Zdim)

        return nt_xent(z, perm, tau=self.tau)
