from typing import Optional

import torch
from torch import nn, Tensor
from torchvision import models

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

WEIGHTS_RELEASE = "v0.1"


def _get_weights_url(rn_model: str, dataset: str, method: str) -> str:
    assert rn_model == "resnet18"
    assert dataset == "eurosat", "Only the 'eurosat' dataset is currently supported"
    METHODS = ["simclr", "moco", "byol", "barlow"]
    assert method in METHODS, f"Expected one of {METHODS}, got {method!r}"
    return f"https://github.com/Pangoraw/GeoSSL/releases/download/{WEIGHTS_RELEASE}/{rn_model}-{dataset}-{method}.pth"


class ResNetBackbone(nn.Module):
    """
    A ResNet model built on top of torchvision.models resnet implementation.
    """

    out_dim: int

    @staticmethod
    def from_pretrained(path: str, progress: bool = True) -> "ResNetBackbone":
        """
        Instanciates the model from pretrained weights.

        >>> model = ResNetBackbone.from_pretrained("resnet18/eurosat/moco")

        Parameters
        ==========
            path: str - an identifier string representing the model.
        Returns
        =======
            backbone: ResNetBackbone - the pretrained backbone.
        """
        rn_model, dataset, method = path.split("/")
        model = ResNetBackbone(rn_model, small_conv=dataset == "eurosat")

        url = _get_weights_url(rn_model, dataset, method)
        state_dict = load_state_dict_from_url(url, progress=progress)
        model.load_state_dict(state_dict)

        return model

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
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
            num_classes=num_classes,
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
