"""Bottleneck(ResNet) module, generator.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""
# pylint: disable=useless-super-delegation
from typing import Union

import torch
from torch import nn as nn

from src.modules.base_generator import GeneratorAbstract
from src.modules.conv import Conv


class Bottleneck(nn.Module):
    """Standard bottleneck block."""

    def __init__(
        self,
        in_channel: int,
        out_channels: int,
        shortcut=True,
        groups: int = 1,
        expansion: float = 0.5,
        activation: Union[str, None] = "ReLU",
    ) -> None:
        """Initialize."""
        super().__init__()
        expansion_channel = int(out_channels * expansion)

        self.conv1 = Conv(in_channel, expansion_channel, 1, 1, activation=activation)
        self.conv2 = Conv(expansion_channel, out_channels, 3, 1, groups=groups)
        self.shortcut = shortcut and in_channel == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        out = self.conv2(self.conv1(x))

        if self.shortcut:
            out = out + x

        return out


class BottleneckGenerator(GeneratorAbstract):
    """Bottleneck block generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self._get_divisible_channel(self.args[0] * self.width_multiply)

    @property
    def base_module(self) -> nn.Module:
        """Returns module class from src.common_modules based on the class name."""
        return getattr(__import__("src.modules", fromlist=[""]), self.name)

    def __call__(self, repeat: int = 1):
        repeat_args = [self.in_channel, self.in_channel, *self.args[1:]]
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        module = []
        if repeat > 1:
            for _ in range(repeat - 1):
                module.append(self.base_module(*repeat_args))
        module.append(self.base_module(*args))
        return self._get_module(module)
