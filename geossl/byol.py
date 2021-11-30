import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .backbones import ResNetBackbone
from .base_method import BaseMethod


class CosineMomentumScheduler:
    def __init__(self, base_momentum, final_momentum, n_steps):
        self.momentum = base_momentum
        self.base_momentum = base_momentum
        self.final_momentum = final_momentum
        self.n_steps = n_steps

    def step(self, step: int):
        self.momentum = (
            self.final_momentum
            - (self.final_momentum - self.base_momentum)
            * (math.cos(math.pi * step / self.n_steps) + 1)
            / 2
        )


class BYOL(BaseMethod):
    def __init__(
        self,
        encoder_s: ResNetBackbone,
        encoder_t: ResNetBackbone,
        base_momentum: float,
        final_momentum: float,
        n_steps: int,
    ):
        super().__init__()

        self.encoder_s = encoder_s
        self.encoder_t = encoder_t

        z_dim = self.encoder_s.out_dim
        build_block = lambda input_dim, hidden_dim, output_dim: nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.projector_s = build_block(z_dim, z_dim, 256)
        self.projector_t = build_block(z_dim, z_dim, 256)
        self.predictor = build_block(256, 512, 256)

        self.momentum_schedule = CosineMomentumScheduler(
            base_momentum, final_momentum, n_steps
        )

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        z1 = self.predictor(self.projector_s(self.encoder_s(x1)))
        z2 = self.predictor(self.projector_s(self.encoder_s(x2)))
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        with torch.no_grad():
            z1p = self.projector_t(self.encoder_t(x1))
            z2p = self.projector_t(self.encoder_t(x2))
            z1p = F.normalize(z1p, dim=-1)
            z2p = F.normalize(z2p, dim=-1)

        return 4 - 2 * (
            F.cosine_similarity(z1, z2p).mean() + F.cosine_similarity(z2, z1p).mean()
        )

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
