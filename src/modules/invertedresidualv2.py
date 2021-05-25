import torch.nn as nn

from src.modules.base_generator import GeneratorAbstract


class InvertedResidualv2(nn.Module):
    """Inverted Residual block Mobilenet V2.
    Reference:
        https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
    """

    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidualv2, self).__init__()
        self.stride = stride
        assert stride in [1, 2], stride

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer)
            )
        layers.extend(
            [
                # dw
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBNReLU(nn.Sequential):
    """Conv-Bn-ReLU used in pytorch official."""

    def __init__(
        self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None
    ):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidualv2Generator(GeneratorAbstract):
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
        """call method.
        InvertedResidualv2 args consists,
        repeat(=n), [c, t, s] // note original notation from paper is [t, c, n, s]
        """
        module = []
        _, t, s = self.args  # c is equivalent as self.out_channel
        inp, oup = self.in_channel, self.out_channel
        for i in range(repeat):
            stride = s if i == 0 else 1
            module.append(
                self.base_module(inp=inp, oup=oup, expand_ratio=t, stride=stride)
            )
            inp = oup
        return self._get_module(module)
