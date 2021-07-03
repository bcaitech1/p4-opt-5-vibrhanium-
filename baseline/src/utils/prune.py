# PRUNING FILTERS FOR EFFICIENT CONVNETS
# https://arxiv.org/pdf/1608.08710.pdf

import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class PrunedVGGNetModule(nn.Module):
    def __init__(
            self,
            out_channels: int = 3,
            pruned_channels: list = []):
        self.model_name = "vgg16"
        self.out_channels = out_channels
        self.pruned_channels = pruned_channels


class PrunedResNetModule(nn.Module):
    def __init__(
            self,
            conv_n_out_channels: list = [3, 3],
            conv_n_pruned_channels: list = [[], []],
            downsample_out_channels: int = 64,
            downsample_pruned_channels: list = [],
    ):
        self.model_name = "resnet34"
        self.conv_n_out_channels = conv_n_out_channels
        self.conv_n_pruned_channels = conv_n_pruned_channels
        self.downsample_out_channels = downsample_out_channels
        self.downsample_pruned_channels = downsample_pruned_channels


class PrunedShuffleNetModule(nn.Module):
    def __init__(
            self,
            out_channels: int = 3,
            pruned_channels: list = [],
            pruned_num: int = 3
    ):
        self.model_name = "shufflenet_v2"
        self.in_channels = out_channels
        self.pruned_channels = pruned_channels
        self.pruned_num = pruned_num


# --for Resnet--
def get_layer_name(module_name: str):
    layer_name = re.compile("layer[0-9]").match(module_name)
    return layer_name.group() if layer_name is not None else None


def setting_layer_name_at_modules(modules, layer_name):
    if not modules.get(layer_name, None):
        modules[layer_name] = [
            [None, None],
            [None, None]
        ]  # [down_conv1, down_bn1, conv1, bn1, conv2, bn2]


def add_in(modules: dict, module: nn.Module, module_name: str):
    layer_name = get_layer_name(module_name)

    if layer_name is not None:  # is_in_layer(layer_name)
        setting_layer_name_at_modules(modules, layer_name)
        if "downsample" in module_name:  # is_downsample
            if isinstance(module, nn.Conv2d):
                modules[layer_name][0] = [module_name, module]
            elif isinstance(module, nn.BatchNorm2d):
                modules[layer_name][1] = [module_name, module]
        else:
            modules[layer_name].append([module_name, module])
    else:
        layer_name = "layer0"
        modules[layer_name].append([module_name, module])


def get_reorder_modules(model: nn.Module):
    reordered_modules = {"layer0": []}
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
            add_in(reordered_modules, module, module_name)
        elif isinstance(module, nn.Linear):
            layer_name = "fc"
            reordered_modules[layer_name] = [[module_name, module]]
            break

    return reordered_modules


def get_in_channels(pruned_module, module_name):
    layer_type = module_name.split(".")[-1]
    if "downsample" in module_name:
        if "0" == layer_type:  # 'conv1'
            in_channels = pruned_module.conv_n_out_channels[1]  # 이전 layer의 conv2
        elif "1" == layer_type:  # 'bn1'
            in_channels = pruned_module.downsample_out_channels  # 현재 layer의 down_conv
    elif layer_type in ["conv1", "bn2"]:
        in_channels = pruned_module.conv_n_out_channels[1]  # conv1: 이전 layer의 conv2 (layer1.conv1일 경우 이전 layer의 conv1)
                                                            # bn2  : 현재 lalyer의 conv2
    elif layer_type in ["conv2", "bn1"]:
        in_channels = pruned_module.conv_n_out_channels[0]  # conv1, bn1: 현재 layer의 conv1
    else:
        raise ValueError(f"{layer_type} in {module_name} is not in ['conv1', 'conv2', 'bn1', 'bn2', '0', '1']")

    return in_channels


def get_pruned_channels(pruned_module, module_name):
    layer_type = module_name.split(".")[-1]
    if "downsample" in module_name:
        if "0" == layer_type:    # 'conv1'
            pruned_module = pruned_module.conv_n_pruned_channels[1]  # 이전 layer의 conv2
        elif "1" == layer_type:  # 'bn1'
            pruned_module = pruned_module.downsample_pruned_channels  # 현재 layer의 down_conv
    elif layer_type in ["conv1", "bn2"]:
        pruned_module = pruned_module.conv_n_pruned_channels[1]     # conv1: 이전 layer의 conv2 (layer1.conv1일 경우 이전 layer의 conv1)
                                                                    # bn2  : 현재 lalyer의 conv2
    elif layer_type in ["conv2", "bn1"]:
        pruned_module = pruned_module.conv_n_pruned_channels[0]     # conv1, bn1: 현재 layer의 conv1
    else:
        raise ValueError(f"{layer_type} in {module_name} is not in ['conv1', 'conv2', 'bn1', 'bn2', '0', '1']")

    return pruned_module


def set_in_channels(pruned_module, module_name, pruned_out_channels):
    layer_type = module_name.split(".")[-1]
    if "downsample" in module_name:
        if "0" == layer_type:  # 'conv1'
            pruned_module.downsample_out_channels = pruned_out_channels  # 현재 layer의 down_conv
    elif layer_type == "conv1":
        pruned_module.conv_n_out_channels[0] = pruned_out_channels
    elif layer_type == "conv2":
        pruned_module.conv_n_out_channels[1] = pruned_out_channels
    else:
        raise ValueError(f"{layer_type} in {module_name} is not in ['conv1', 'conv2', '0']")


def set_pruned_channels(pruned_module, module_name, pruned_channels=None):
    layer_type = module_name.split(".")[-1]
    if "downsample" in module_name:
        if "0" == layer_type:  # 'conv1'
            pruned_module.downsample_pruned_channels = pruned_channels  # 현재 layer의 down_conv
    elif layer_type == "conv1":
        pruned_module.conv_n_pruned_channels[0] = pruned_channels
    elif layer_type == "conv2":
        pruned_module.conv_n_pruned_channels[1] = pruned_channels
    else:
        raise ValueError(f"{layer_type} in {module_name} is not in ['conv1', 'conv2', '0']")

    return pruned_module


def is_add_with_downsample(layer_name: str, i: int):
    return True if layer_name in ["layer2", "layer3", "layer4"] and i == 4 else False
# ----


def replace_module(model: nn.Module, module_name: str, module):
    digit = re.compile("[0-9]")

    layers_info, layer_type = module_name.split(".")[:-1], module_name.split(".")[-1]
    prev_module = model
    for layer_name in layers_info:
        if digit.match(layer_name):
            prev_module = prev_module[int(layer_name)]
        else:
            prev_module = getattr(prev_module, layer_name)
    setattr(prev_module, layer_type, module)


def get_pruned_in_channels_and_pruned_channels(module_name: str, pruned_module):
    pruned_in_channels, pruned_channels = None, None
    if pruned_module.model_name in ["vgg16", "shufflenet_v2"]:
        pruned_in_channels = pruned_module.out_channels
        pruned_channels = pruned_module.pruned_channels

    elif pruned_module.model_name == "resnet34":
        if "layer1.0.conv1" in module_name:  # layer1.0.conv1일 경우 이전 layer의 conv1 참조
            pruned_in_channels = pruned_module.conv_n_out_channels[0]
            pruned_channels = pruned_module.conv_n_pruned_channels[0]
        else:  # 아닐 경우, conv1인 경우 이전 layer의 conv2 참조, conv2인 경우 이전 layer의 conv1 참조
            pruned_in_channels = get_in_channels(pruned_module, module_name)
            pruned_channels = get_pruned_channels(pruned_module, module_name)

    return pruned_in_channels, pruned_channels


def get_conv_params(conv_module: nn.Conv2d):
    in_channels, out_channels, kernel_size, stride, padding = (
        conv_module.in_channels,
        conv_module.out_channels,
        conv_module.kernel_size,
        conv_module.stride,
        conv_module.padding,
    )
    bias = False if conv_module.bias is None else True

    return in_channels, out_channels, kernel_size, stride, padding, bias


def get_bn_params(bn_module: nn.BatchNorm2d):
    eps, momentum, affine, track_running_stats = (
        bn_module.eps,
        bn_module.momentum,
        bn_module.affine,
        bn_module.track_running_stats,
    )

    return eps, momentum, affine, track_running_stats


def get_linear_params(linear_module: nn.Linear):
    in_features, out_features = linear_module.in_features, linear_module.out_features
    bias = False if linear_module.bias is None else True

    return in_features, out_features, bias


def get_pruned_module(module_name: str, module: nn.Module, pruned_module):
    pruned_in_channels, pruned_channels, axis = None, None, None
    if isinstance(module, nn.Conv2d):
        _, out_channels, kernel_size, stride, padding, bias = get_conv_params(module)
        pruned_in_channels, pruned_channels = get_pruned_in_channels_and_pruned_channels(module_name, pruned_module)
        axis = 1

        pruned_module = nn.Conv2d(
            in_channels=pruned_in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
    elif isinstance(module, nn.BatchNorm2d):
        eps, momentum, affine, track_running_stats = get_bn_params(module)
        pruned_in_channels = get_in_channels(pruned_module, module_name)
        pruned_channels = get_pruned_channels(pruned_module, module_name)
        axis = 0

        pruned_module = nn.BatchNorm2d(
            num_features=pruned_in_channels,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )
    elif isinstance(module, nn.Linear):
        _, out_features, bias = get_linear_params(module)
        axis = 1
        if pruned_module.model_name == "vgg16":
            pruned_in_channels = pruned_module.out_channels * 49
        elif pruned_module.model_name == "resnet34":
            pruned_in_channels = pruned_module.conv_n_out_channels[1]
            pruned_channels = pruned_module.conv_n_pruned_channels[1]

        pruned_module = nn.Linear(
            in_features=pruned_in_channels,
            out_features=out_features,
            bias=bias
        )

    return pruned_module, pruned_channels, axis


def get_module_pruning_in_channels(module_name: str, module: nn.Module, pruned_module):
    """이전 layer에서 prune된 channels 처리"""
    # get pruned module
    pruned_module, pruned_channels, axis = get_pruned_module(module_name, module, pruned_module)

    # pruned weight
    pruned_weight = module.weight.cpu().numpy()
    for pruned_channel_num, channel in enumerate(pruned_channels):
        pruned_weight = np.delete(pruned_weight, channel - pruned_channel_num, axis=axis)

    # set pruned weight at pruned module
    pruned_weight = torch.from_numpy(pruned_weight)
    pruned_module.weight = torch.nn.Parameter(
        pruned_weight, requires_grad=True
    )

    return pruned_module


def delete_pruned_channels(pruned_weight: np.array):
    """현재 layer에서 pruning된 channels 삭제"""
    pruned_channel_num = 0
    pruned_channels = []
    for channel, weight in enumerate(pruned_weight):
        if np.array_equal(weight, np.zeros_like(weight)):
            pruned_weight = np.delete(pruned_weight, channel - pruned_channel_num, axis=0)
            pruned_channel_num += 1
            pruned_channels.append(channel)

    return pruned_weight, pruned_channel_num, pruned_channels


def get_pruned_conv_module(module: nn.Conv2d, pruned_out_channels: int, pruned_weight: np.array):
    """pruning된 filter를 적용한 새로운(pruned) conv2 생성"""
    in_channels, _, kernel_size, stride, padding, bias = get_conv_params(module)
    pruned_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=pruned_out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    pruned_conv.weight = torch.nn.Parameter(pruned_weight, requires_grad=True)

    return pruned_conv


@torch.no_grad()
def filter_prune_for_vgg16(
        model: nn.Module, conv_prun_norm: int = 2, conv_prun_rate: float = 0.3,
):
    pruned_module = PrunedVGGNetModule()

    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            pruned_conv = get_module_pruning_in_channels(module_name, module, pruned_module)

            # pruning n 개 prune (값이 0으로 변경)
            pruning_container = prune.LnStructured.apply(
                pruned_conv,
                name="weight",
                amount=conv_prun_rate,
                n=conv_prun_norm,
                dim=0,
            )
            pruned_weight = pruning_container.prune(pruned_conv.weight)

            # prune 된 (모든 값이 0인) out channel 삭제
            pruned_weight, pruned_channel_num, pruned_channels = delete_pruned_channels(pruned_weight)
            pruned_out_channels = module.out_channels - pruned_channel_num

            # setting
            pruned_module.out_channels = pruned_out_channels
            pruned_module.pruned_channels = pruned_channels

            # replace module
            pruned_conv = get_pruned_conv_module(pruned_conv, pruned_out_channels, pruned_weight)
            replace_module(model, module_name, pruned_conv)

        elif isinstance(module, nn.BatchNorm2d):
            pruned_bn = get_module_pruning_in_channels(module_name, module, pruned_module)  # pruning num_features
            replace_module(model, module_name, pruned_bn)

        elif isinstance(module, nn.Linear):
            pruned_linear, _, _ = get_pruned_module(module_name, module, pruned_module)
            replace_module(model, module_name, pruned_linear)
            break

    return model


@torch.no_grad()
def filter_prune_for_resnet34(
        model: nn.Module, conv_prun_norm: int = 2, conv_prun_rate: float = 0.3,
):
    pruned_module = PrunedResNetModule()
    reordered_modules = get_reorder_modules(model)

    for layer_name, modules in reordered_modules.items():
        for i, (module_name, module) in enumerate(modules):
            if isinstance(module, nn.Conv2d):
                pruned_conv = get_module_pruning_in_channels(module_name, module, pruned_module)

                if is_add_with_downsample(layer_name, i):
                    # downsample에서의 pruning 결과 그대로 적용
                    pruned_out_channels = pruned_module.downsample_out_channels
                    pruned_channels = pruned_module.downsample_pruned_channels
                    pruned_weight = pruned_conv.weight

                    for t, channel in enumerate(pruned_channels):
                        pruned_weight = np.delete(pruned_weight, channel - t, axis=0)
                else:
                    # prune (값이 0으로 변경)
                    pruning_container = prune.LnStructured.apply(
                        pruned_conv,
                        name="weight",
                        amount=conv_prun_rate,
                        n=conv_prun_norm,
                        dim=0,
                    )
                    pruned_weight = pruning_container.prune(pruned_conv.weight)

                    # prune 된 (모든 값이 0인) out channel 삭제
                    pruned_weight, pruned_channel_num, pruned_channels = delete_pruned_channels(pruned_weight)
                    pruned_out_channels = module.out_channels - pruned_channel_num

                # setting
                set_in_channels(pruned_module, module_name, pruned_out_channels)
                set_pruned_channels(pruned_module, module_name, pruned_channels)

                # replace module
                pruned_conv = get_pruned_conv_module(pruned_conv, pruned_out_channels, pruned_weight)
                replace_module(model, module_name, pruned_conv)

            elif isinstance(module, nn.BatchNorm2d):
                pruned_bn = get_module_pruning_in_channels(module_name, module, pruned_module)  # pruning num_features
                replace_module(model, module_name, pruned_bn)

            elif isinstance(module, nn.Linear):
                pruned_linear = get_module_pruning_in_channels(module_name, module, pruned_module)
                replace_module(model, module_name, pruned_linear)

    return model


@torch.no_grad()
def filter_prune_for_shufflenet_v2(
        model: nn.Module, conv_prun_norm: int = 1, conv_prun_rate: float = 0.3,
):
    pruned_module = PrunedShuffleNetModule()

    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            pruned_conv = get_module_pruning_in_channels(module_name, module, pruned_module)

            # 1X1 CONV(PRUNING 적용 + PRUNING) -> DWCONV(PRING 적용만) -> 1X1 CONV(PRUNING 적용 + PRUNING)
            if "conv1.0" in module_name:
                conv_prun_num = int(out_channels * conv_prun_rate)
                out_channels = out_channels
            else:
                out_channels = in_channels

            # stage2 + stage2 (concat) = stage3
            if "stage3.0.branch1.0" in module_name or "stage4.0.branch1.0" in module_name:
                in_channels = pruned_module.out_channels * 2
                pruned_module.pruned_num = pruned_module.pruned_num * 2
            conv_prun_num = pruned_module.pruned_num

            pruned_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=in_channels,
            )
            if groups == 1:
                pruned_weight = pruned_conv.weight
                for t, channel in enumerate(pruned_channels):
                    pruned_weight = np.delete(pruned_weight, channel - t, axis=1)

            pruned_weight = torch.from_numpy(pruned_weight)
            pruned_conv.weight = torch.nn.Parameter(pruned_weight, requires_grad=True)

            # pruning n 개 prune (값이 0으로 변경)
            pruning_container = prune.LnStructured.apply(
                pruned_conv,
                name="weight",
                amount=conv_prun_num,
                n=conv_prun_norm,
                dim=0,
            )
            pruned_weight = pruning_container.prune(pruned_weight)

            # pruning되지 않은 kernel 추출
            new_out_channels = 0
            new_weight = []
            pruned_channels = []
            for channel, kernel in enumerate(pruned_weight):
                if np.array_equal(kernel, np.zeros_like(kernel)):
                    pruned_channels.append(channel)
                else:
                    new_out_channels += 1
                    new_weight.append(kernel)
            new_weight = torch.stack(new_weight, dim=0)
            pruned_module.in_channels = new_out_channels
            pruned_module.pruned_channels = pruned_channels

            if groups == 1:
                new_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=new_out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )

            else:
                new_conv = nn.Conv2d(
                    in_channels=new_out_channels,
                    out_channels=new_out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=new_out_channels,
                )
            new_conv.weight = torch.nn.Parameter(new_weight, requires_grad=True)

            if module_name == "conv5.0":
                new_conv = nn.Conv2d(
                    in_channels=144,
                    out_channels=1024,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )

            replace_module(model, module_name, new_conv)

        elif isinstance(module, nn.BatchNorm2d):
            pruned_bn = get_module_pruning_in_channels(module_name, module, pruned_module)  # pruning num_features
            replace_module(model, module_name, pruned_bn)

        elif isinstance(module, nn.Linear):
            pruned_linear = get_module_pruning_in_channels(module_name, module, pruned_module)
            replace_module(model, module_name, pruned_linear)

    return model


if __name__ == "__main__":
    from os import path
    import sys

    import torch
    import torchvision

    ROOT_PATH = "/opt/ml/"

    BASE_PATH = path.join(ROOT_PATH, "code")
    sys.path.append(BASE_PATH)

    # -- 모델 선언
    num_classes = 9

    # model = torchvision.models.vgg16(pretrained=False)
    # model.classifier[6] = nn.Linear(4096, num_classes)
    # optimized_model = filter_prune_for_vgg16(model)

    model = torchvision.models.resnet34(pretrained=False)
    model.fc = nn.Linear(512, num_classes)
    optimized_model = filter_prune_for_resnet34(model)

    # model = torchvision.models.shufflenet_v2_x0_5(pretrained=False)
    # model.fc = nn.Linear(1024, num_classes)
    # optimized_model = filter_prune_for_sheufflenet_v2(model)

    print(optimized_model)

    test_data = torch.rand(1, 3, 224, 224)
    output = optimized_model(test_data)
    print(output.shape)

    # out_path = "/opt/ml/output/pruned_test.pt"
    # torch.save(model=optimized_model, f=out_path)
    # model = torch.load(out_path)
    # print(model)
