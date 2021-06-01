import torch
from torch import nn as nn
from torch.nn.functional import dropout

from src.modules.base_generator import GeneratorAbstract


class Dropout(nn.Module):
    """Dropout module."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        """
        Args:
            p – probability of an element to be zeroed. Default: 0.5
            training – apply dropout if is True. Default: True
            inplace – If set to True, will do this operation in-place. Default: False
        """
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return dropout(x, self.p, self.training, self.inplace)


class DropoutGenerator(GeneratorAbstract):
    """Dropout module generator for parsing."""

    def __init__(self, *args, **kwargs):
        """Initailize."""
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self.in_channel

    def __call__(self, repeat: int = 1):
        return self._get_module(Dropout(*self.args))
