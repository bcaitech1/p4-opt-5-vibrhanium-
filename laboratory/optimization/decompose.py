from typing import List

import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import partial_tucker


# def set_register_buffer(module: nn.Module):
#     for name, param in module.named_modules():
#         if isinstance(param, nn.Conv2d):
#             param.register_buffer('rank', torch.Tensor([0.5, 0.5])) # rank in, out                                               


def tucker_decomposition_conv_layer(
    layer: nn.Module,
    normed_rank: List[int] = [0.5, 0.5],
):
    tl.set_backend('pytorch')

    # if hasattr(layer, "rank"):
    #     normed_rank = getattr(layer, "rank")
    rank = [int(r * layer.weight.shape[i]) for i, r in enumerate(normed_rank)]
    rank = [max(r, 2) for r in rank]

    core, [last, first] = partial_tucker(
        layer.weight.data,
        modes=[0, 1],
        n_iter_max=20000000,
        rank=rank,
        init="svd"
    )

    first_layer = nn.Conv2d(
        in_channels=first.shape[0],
        out_channels=first.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False
    )

    core_layer = nn.Conv2d(
        in_channels=core.shape[1],
        out_channels=core.shape[0],
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=False
    )

    last_layer = nn.Conv2d(
        in_channels=last.shape[1],
        out_channels=last.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=True
    )

    if hasattr(layer, "bias") and layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    first_layer.weight.data = (torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1))
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)


def decompose(module: nn.Module):
    for name, child_layer in module.named_children():
        if isinstance(child_layer, nn.Conv2d):
            if child_layer.kernel_size[0] != 1 and child_layer.groups == 1:
                # print(f"{(name)}: {child_layer} 분해 중..")
                setattr(module, name, tucker_decomposition_conv_layer(child_layer))
                # print("분해 완료!")
        elif list(child_layer.children()):
            decomposed_module = decompose(child_layer)
            if decomposed_module:
                setattr(module, name, decomposed_module)
    
    return module
