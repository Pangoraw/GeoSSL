from torch import nn, Tensor


class BaseMethod(nn.Module):
    r"""
    Base class for all methods
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y1: Tensor, y2: Tensor) -> Tensor:
        raise NotImplementedError()

    def step(self, step: int):
        pass
