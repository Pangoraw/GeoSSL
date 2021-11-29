from typing import Optional

import torch
from torch import nn, Tensor
from torchvision import models


class ResNetBackbone(nn.Module):
    out_dim: int

    def __init__(self, resnet: str, weights_path: Optional[str]=None):
        super(ResNetBackbone, self).__init__()
        self.model = models.__dict__[resnet](pretrained=False)

        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)

        self.out_dim = self.model.fc.in_features
        self.model.fc = nn.Identity() # remove the last layer

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
