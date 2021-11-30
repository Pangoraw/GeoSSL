from typing import List
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .base_method import BaseMethod
from .backbones import ResNetBackbone
from .byol import CosineMomentumScheduler


class MoCo(BaseMethod):
    queue: Tensor

    def __init__(
        self,
        encoder_s: ResNetBackbone,
        encoder_t: ResNetBackbone,
        feat_dim: int,
        base_momentum: float,
        final_momentum: float,
        n_steps: int,
        queue_size: int,
    ):
        super().__init__()
        self.encoder_s = encoder_s
        self.encoder_t = encoder_t

        z_dim = self.encoder_s.out_dim
        build_projector = lambda: nn.Sequential(
            nn.Linear(z_dim, z_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim, feat_dim, bias=False),
        )

        self.projector_s = build_projector()
        self.projector_t = build_projector()
        self.queue_size = queue_size
        self.register_buffer("queue_end", torch.tensor(0))
        self.register_buffer("queue", torch.zeros((queue_size, feat_dim)))
        self.queue = F.normalize(self.queue, dim=-1)
        self.tau = 0.2
        self.momentum_schedule = CosineMomentumScheduler(
            base_momentum, final_momentum, n_steps
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        z1 = self.projector_s(self.encoder_s(x1))
        z1 = F.normalize(z1, dim=-1)
        bs = z1.size(0)

        with torch.no_grad():
            z2 = self.projector_t(self.encoder_t(x2))
            z2 = F.normalize(z2, dim=-1)

            # Make room for new batch
            self.queue = torch.roll(self.queue, bs, 0)
            self.queue[:bs, :] = z2

            self.queue_end = torch.min(
                self.queue_end + bs, torch.tensor(self.queue_size)
            )

        sim = (z1 @ self.queue[: self.queue_end, :].T) / self.tau
        perm = torch.arange(0, bs, device=sim.device)

        return F.cross_entropy(sim, perm,)

    @torch.no_grad()
    def step(self, step: int):
        super().step(step)
        m = self.momentum_schedule.momentum
        self.momentum_schedule.step(step)
        for param_s, param_t in zip(
            self.encoder_s.parameters(), self.encoder_t.parameters()
        ):
            param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)
        for param_s, param_t in zip(
            self.projector_s.parameters(), self.projector_t.parameters()
        ):
            param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)
