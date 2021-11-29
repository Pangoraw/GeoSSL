import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .backbones import ResNetBackbone


# Should we incorporate changes proposed in SimCLRv2?
# this would mean that the backbone has a different architecture (one more projection layer) and
# is not directly comparable with other benchmarks.
class SimCLR(nn.Module):
    """
    An implementation of SimCLR[1].

    [1]T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A Simple Framework for Contrastive Learning of Visual Representations,”
       arXiv:2002.05709 [cs, stat], Jun. 2020, Accessed: Nov. 02, 2021. [Online]. Available: http://arxiv.org/abs/2002.05709
    """

    def __init__(self, backbone: ResNetBackbone, tau: float):
        super(SimCLR, self).__init__()
        self.backbone = backbone

        z_dim = h_dim = backbone.out_dim
        self.projection_head = nn.Sequential(
            nn.Linear(h_dim, z_dim, bias=False),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim, bias=False),
        )
        self.tau = tau

    def loss(self, z, perm, tau: float) -> Tensor:
        """Pairwise NT-Xent loss
        Parameters
        ==========
        p: Tensor - the projected features
        perm: Tensor - the positive pairs permutations in p

        Returns
        =======
        loss: Tensor - the NT-Xent loss
        """
        two_b = z.size(0)

        features = F.normalize(z, dim=1)
        sim = features @ features.T # sim.shape == (2b, 2b)

        mask = ~torch.eye(two_b, device=sim.device, dtype=torch.bool) # set i,i sim to 0
        sim = (mask * sim) / tau # (2b, 2b)

        return F.cross_entropy(sim, perm)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor: # x1.shape == x2.shape == (b, c, h, w)
        """
        Returns the pairwise NT-Xent loss between x1 and x2
        Parameters
        ==========
            x1: Tensor - random augmentation of x
            x2: Tensor - random augmentation of x
        Returns
        =======
            loss: Tensor - the NT-Xent loss
        """
        b = x1.size(0)
        xp = torch.cat((x1, x2)) # xp.shape == (2b, c, h, w)

        # match each x1 with its corresponding x2
        perm = torch.cat((torch.arange(b) + b, torch.arange(b)), dim=0).to(xp.device)

        h = self.backbone(xp) # h.shape == (2b, C, H, W)
        z = self.projection_head(h) # p.shape == (2b, C)
        return self.loss(z, perm, tau=self.tau)
