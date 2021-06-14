"""PyTorch Module and ModuleGenerator."""

from src.shufflenet import Shufflenet_v205
from src.squeezenet import Squeezenet1_1
from src.resnet import Resnet34

__all__ = [
    "Shufflenet_v205",
    "Squeezenet1_1",
    "Resnet34",
]
