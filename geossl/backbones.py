from typing import Optional

import torch
from torch import nn, Tensor
from torchvision import models


class ResNetBackbone(nn.Module):
    """
    A ResNet model built on top of torchvision.models resnet implementation.
    """

    out_dim: int

    def __init__(
        self,
        resnet: str,
        num_classes: int = 10,
        with_classifier: bool = False,
        weights_path: Optional[str] = None,
        pretrained: bool = False,
        small_conv: bool = False,
    ):
        super(ResNetBackbone, self).__init__()
        self.model = models.__dict__[resnet](
            pretrained=pretrained, num_classes=num_classes
        )

        if weights_path is not None and pretrained:
            raise Exception("Can't use both pretrained and weights_path")
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)

        if not with_classifier:
            self.out_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()  # remove the last layer
        else:
            self.out_dim = num_classes

        if small_conv:
            # Change the filter size to 3x3 for small input images
            # and remove maxpooling op.
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            self.model.maxpool = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
