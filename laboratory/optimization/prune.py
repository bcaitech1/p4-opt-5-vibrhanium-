import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


class PrunedModule(nn.Module):
    # conv1 -> bn1 -> layer1(downsample_conv -> downsample_bn -> conv1 -> conv2)
    # downsample_conv -> downsample_bn -> conv1 -> conv2  -> downsample_conv [1] -> downsample_bn -> conv1 [1] -> conv2 [0] -> ...
    def __init__(
        self,
        conv_n_in_channels: list = [3, 3],
        conv_n_pruned_channels: list = [[], []],
        downsample_out_channels: int = 64,
        downsample_pruned_channels: list = [],
    ):
        self.conv_n_in_channels = conv_n_in_channels
        self.conv_n_pruned_channels = conv_n_pruned_channels
        self.downsample_out_channels = downsample_out_channels
        self.downsample_pruned_channels = downsample_pruned_channels


class PrunedModuleShffleNet(nn.Module):
    # conv1 -> bn1 -> layer1(downsample_conv -> downsample_bn -> conv1 -> conv2)
    # downsample_conv -> downsample_bn -> conv1 -> conv2  -> downsample_conv [1] -> downsample_bn -> conv1 [1] -> conv2 [0] -> ...
    def __init__(
        self, in_channels: int = 3, pruned_channels: list = [], pruned_num: int = 3
    ):
        self.in_channels = in_channels
        self.pruned_channels = pruned_channels
        self.pruned_num = pruned_num


class PrunedModule_vgg(nn.Module):
    def __init__(self, in_channels: int = 3, pruned_channels: list = []):
        self.in_channels = in_channels
        self.pruned_channels = pruned_channels


def set_module(model: nn.Module, name: str, new_module):
    p = re.compile("[0-9]")

    layers_name, module_name = name.split(".")[:-1], name.split(".")[-1]
    prev_module = model
    for layer_name in layers_name:
        if p.match(layer_name):
            prev_module = prev_module[int(layer_name)]
        else:
            prev_module = getattr(prev_module, layer_name)
    setattr(prev_module, module_name, new_module)


def set_layers_dict(
    model: nn.Module, name: str, layers_dict
):  # layers_dict = {'layer0': [], 'layer1': [], ... 'fc': []}
    p_num = re.compile("[0-9]")
    # 'layer1.0.conv1' -> model.layer1[0].conv1
    layers_name, module_name = name.split(".")[:-1], name.split(".")[-1]
    module = model
    for layer_name in layers_name:
        if p_num.match(layer_name):
            module = module[int(layer_name)]
        else:
            module = getattr(module, layer_name)
    # "layer1": down -> conv1 -> bn1 -> conv2 -> bn2,
    p_layer = re.compile("layer[0-9]")
    m_layer = p_layer.match(name)
    if m_layer:
        layer_n = m_layer.group()
        if not layers_dict.get(layer_n, None):
            layers_dict[layer_n] = [
                None,
                None,
            ]  # [down_conv1, down_bn1, conv1, bn1, conv2, bn2,]
        if "downsample" in name:
            if "0" in module_name:
                layers_dict[layer_n][0] = [
                    module,
                    module_name,
                    name,
                ]  # 'module': 'layer1.0'의 module object module_name = 'conv1', name = 'layer1.0.conv1'
            elif "1" in module_name:
                layers_dict[layer_n][1] = [module, module_name, name]
        elif "conv1" in name:
            layers_dict[layer_n].append([module, module_name, name])
        elif "bn1" in name:
            layers_dict[layer_n].append([module, module_name, name])
        elif "conv2" in name:
            layers_dict[layer_n].append([module, module_name, name])
        elif "bn2" in name:
            layers_dict[layer_n].append([module, module_name, name])
        else:
            print(f"{name} is not pruned")

    elif name == "fc":
        if not layers_dict.get(name, None):
            layers_dict[name] = []
        layers_dict["fc"].append([module, module_name, name])
    else:
        layers_dict["layer0"].append([module, module_name, name])


def get_in_channels(pruned_module, name):
    if "downsample" in name:
        temp_name = name.split(".")[-1]
        if "0" == temp_name:  # 'conv1'
            in_channels = pruned_module.conv_n_in_channels[1]  # 이전 layer의 conv2
        elif "1" == temp_name:  # 'bn1'
            in_channels = pruned_module.downsample_out_channels  # 현재 layer의 down_conv
    elif "conv1" in name:
        in_channels = pruned_module.conv_n_in_channels[
            1
        ]  # 이전 layer의 conv2 (layer1.conv1일 경우 이전 layer의 conv1)
    elif "conv2" in name:
        in_channels = pruned_module.conv_n_in_channels[0]  # 현재 layer의 conv1
    elif "bn1" in name:
        in_channels = pruned_module.conv_n_in_channels[0]  # 현재 layer의 conv1
    elif "bn2" in name:
        in_channels = pruned_module.conv_n_in_channels[1]  # 현재 alyer의 conv2
    else:
        print("*error*", name)

    return in_channels


def get_pruned_channels(pruned_module, name):
    # 현재 layer에서 pruning된 channel 처리
    if "downsample" in name:
        temp_name = name.split(".")[-1]
        if "0" == temp_name:  # 'conv1'
            pruned_module = pruned_module.conv_n_pruned_channels[1]  # 이전 layer의 conv2
        elif "1" == temp_name:  # 'bn1'
            pruned_module = (
                pruned_module.downsample_pruned_channels
            )  # 현재 layer의 down_conv
    elif "conv1" in name:
        pruned_module = pruned_module.conv_n_pruned_channels[
            1
        ]  # 이전 layer의 conv2 (layer1.conv1일 경우 이전 layer의 conv1)
    elif "conv2" in name:
        pruned_module = pruned_module.conv_n_pruned_channels[0]  # 현재 layer의 conv1
    elif "bn1" in name:
        pruned_module = pruned_module.conv_n_pruned_channels[0]  # 현재 layer의 conv1
    elif "bn2" in name:
        pruned_module = pruned_module.conv_n_pruned_channels[1]  # 현재 alyer의 conv2
    else:
        print("*error*", name)

    return pruned_module


def set_in_channels(pruned_module, name, new_out_channels):
    # 현재 layer에서 pruning된 channel 처리
    if "downsample" in name:
        temp_name = name.split(".")[-1]
        if "0" == temp_name:  # 'conv1'
            pruned_module.downsample_out_channels = (
                new_out_channels  # 현재 layer의 down_conv
            )
    elif "conv1" in name:
        pruned_module.conv_n_in_channels[0] = new_out_channels
    elif "conv2" in name:
        pruned_module.conv_n_in_channels[1] = new_out_channels
    else:
        print("*error*", name)


def set_pruned_channels(pruned_module, name, new_pruned_channels=None):
    # 현재 layer에서 pruning된 channel 처리
    if "downsample" in name:
        temp_name = name.split(".")[-1]
        if "0" == temp_name:  # 'conv1'
            pruned_module.downsample_pruned_channels = (
                new_pruned_channels  # 현재 layer의 down_conv
            )
    elif "conv1" in name:
        pruned_module.conv_n_pruned_channels[0] = new_pruned_channels
    elif "conv2" in name:
        pruned_module.conv_n_pruned_channels[1] = new_pruned_channels
    else:
        print("*error*", name)

    return pruned_module


def model_prune(
    model: nn.Module,
    conv_prun_norm: int = 1,
    conv_prun_rate: float = 0.2,
    linear_prun_norm: int = 1,
    linear_prun_rate: float = 0.4,
    structured: bool = False,
):

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            if structured:
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=conv_prun_rate,
                    n=conv_prun_norm,
                    dim=0,
                )
            else:
                if conv_prun_norm == 1:
                    prune.l1_unstructured(module, name="weight", amount=conv_prun_rate)
                else:
                    prune.l2_unstructured(module, name="weight", amount=conv_prun_rate)
            prune.remove(module, "weight")

        elif isinstance(module, torch.nn.modules.linear.Linear):
            if structured:
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=linear_prun_rate,
                    n=conv_prun_norm,
                    dim=0,
                )
            else:
                if linear_prun_norm == 1:
                    prune.l1_unstructured(
                        module, name="weight", amount=linear_prun_rate
                    )
                else:
                    prune.l2_unstructured(
                        module, name="weight", amount=linear_prun_rate
                    )

            prune.remove(module, "weight")

    print("-" * 10 + "prun 적용 모듈" + "-" * 10)
    print(dict(model.named_buffers()).keys())

    return model


@torch.no_grad()
def model_structured_prune_for_shufflenet_v2(
    model: nn.Module, conv_prun_norm: int = 1, conv_prun_rate: float = 0.3,
):
    # PRUNING FILTERS FOR EFFICIENT CONVNETS
    # https://arxiv.org/pdf/1608.08710.pdf

    pruned_module = PrunedModuleShffleNet()

    # layer 별로 name 추출
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # setting
            out_channels, kernel_size, stride, padding, groups = (
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.groups,
            )
            bias = False if module.bias == None else True

            in_channels = pruned_module.in_channels
            pruned_channels = pruned_module.pruned_channels
            pruned_weight = module.weight.cpu().numpy()
            # 1X1 CONV(PRUNING 적용 + PRUNING) -> DWCONV(PRING 적용(IN, OUT)만) -> 1X1 CONV(PRUNING 적용(IN, OUT) + PRUNING)
            # 이전 layer에서 pruning된 channel(in_channels) 처리

            if name == "model.conv1.0":
                conv_prun_num = int(out_channels * conv_prun_rate)
                if conv_prun_num % 2 == 1:
                    conv_prun_num = conv_prun_num - 1
                else:
                    conv_prun_num = conv_prun_num
                pruned_module.pruned_num = conv_prun_num
                out_channels = out_channels
            else:
                out_channels = in_channels

            if name in [
                "model.stage3.0.branch1.0",
                "model.stage4.0.branch1.0",
            ]:
                pruned_module.pruned_num = pruned_module.pruned_num * 2
            conv_prun_num = pruned_module.pruned_num

            if name == "model.conv5.0":
                print(name)

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
                # for t, channel in enumerate(pruned_channels):
                #     pruned_weight = np.delete(pruned_weight, channel - t, axis=1)
                pruned_weight = pruned_weight[
                    :,
                    np.array(
                        list(
                            set(range(pruned_weight.shape[1]))
                            - set(pruned_module.pruned_channels)
                        )
                    ),
                    :,
                    :,
                ]

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

            if name == "model.conv5.0":
                new_conv = nn.Conv2d(
                    in_channels=144,
                    out_channels=1024,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )

            set_module(model, name, new_conv)

        elif isinstance(module, nn.BatchNorm2d):
            # setting
            eps, momentum, affine, track_running_stats = (
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
            )
            in_channels = pruned_module.in_channels
            pruned_channels = pruned_module.pruned_channels

            # 이전 layer에서 pruning된 channel(in_channels) 처리
            pruned_weight = module.weight.cpu().numpy()
            for t, channel in enumerate(pruned_channels):
                pruned_weight = np.delete(pruned_weight, channel - t, axis=0)
            pruned_weight = torch.from_numpy(pruned_weight)
            module.weight = torch.nn.Parameter(pruned_weight, requires_grad=True)

            new_batchnorm = nn.BatchNorm2d(
                in_channels, eps, momentum, affine, track_running_stats
            )  # num_features

            if name == "model.conv5.1":
                new_batchnorm = nn.BatchNorm2d(
                    1024, eps, momentum, affine, track_running_stats
                )  # num_features
            set_module(model, name, new_batchnorm)

        elif isinstance(module, nn.Linear):
            out_features = module.out_features
            bias = False if module.bias == None else True
            in_channels = pruned_module.in_channels
            new_linear = nn.Linear(
                in_features=in_channels, out_features=out_features, bias=bias
            )  # in_features

            if name == "model.fc":
                new_linear = nn.Linear(
                    in_features=1024, out_features=9, bias=bias
                )  # in_features
            set_module(model, name, new_linear)

    return model


@torch.no_grad()
def model_structured_prune_for_resnet34(
    model: nn.Module, conv_prun_norm: int = 2, conv_prun_rate: float = 0.3,
):
    # PRUNING FILTERS FOR EFFICIENT CONVNETS
    # https://arxiv.org/pdf/1608.08710.pdf
    pruned_module = PrunedModule()
    # layer 별로 name 추출
    layers_dict = {"layer0": []}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            set_layers_dict(model, name, layers_dict)
        elif isinstance(module, nn.BatchNorm2d):
            set_layers_dict(model, name, layers_dict)
        elif isinstance(module, nn.Linear):
            set_layers_dict(model, name, layers_dict)
    p_num = re.compile("[0-9]")
    for layer_n, module_infos in layers_dict.items():
        for i, module_info in enumerate(module_infos):  # layer
            if module_info:  # downsample, conv1, bn1, Conv2, ...
                prev_module, module_name, name = module_info
            else:
                continue
            if p_num.match(module_name):
                module = prev_module[int(module_name)]
            else:
                module = getattr(prev_module, module_name)
            if isinstance(module, nn.Conv2d):
                # setting
                out_channels, kernel_size, stride, padding = (
                    module.out_channels,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                )
                bias = False if module.bias == None else True
                if name == "layer1.0.conv1":  # layer1.0.conv1일 경우 이전 layer의 conv1
                    in_channels = pruned_module.conv_n_in_channels[0]
                    pruned_channels = pruned_module.conv_n_pruned_channels[0]
                else:
                    in_channels = get_in_channels(pruned_module, name)
                    pruned_channels = get_pruned_channels(pruned_module, name)
                # 이전 layer에서 pruning된 channel(in_channels) 처리
                pruned_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
                pruned_weight = module.weight.cpu().numpy()
                for t, channel in enumerate(pruned_channels):
                    pruned_weight = np.delete(pruned_weight, channel - t, axis=1)
                pruned_weight = torch.from_numpy(pruned_weight)
                # module.weight = torch.nn.Parameter(pruned_weight,requires_grad=True)
                pruned_conv.weight = torch.nn.Parameter(
                    pruned_weight, requires_grad=True
                )
                if (
                    layer_n in ["layer2", "layer3", "layer4"] and i == 4
                ):  # downsample 수행한 경우 pruning 진행 X, downsample의 값 가져오기
                    new_out_channels = pruned_module.downsample_out_channels
                    pruned_channels = pruned_module.downsample_pruned_channels
                    for t, channel in enumerate(pruned_channels):
                        pruned_weight = np.delete(pruned_weight, channel - t, axis=0)
                    new_weight = pruned_weight
                else:
                    # pruning n 개 prune (값이 0으로 변경)
                    pruning_container = prune.LnStructured.apply(
                        pruned_conv,
                        name="weight",
                        amount=conv_prun_rate,
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
                set_in_channels(pruned_module, name, new_out_channels)
                set_pruned_channels(pruned_module, name, pruned_channels)
                new_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=new_out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
                new_conv.weight = torch.nn.Parameter(new_weight, requires_grad=True)
                # set_module(model, name, new_conv)
                setattr(prev_module, module_name, new_conv)
            elif isinstance(module, nn.BatchNorm2d):
                # setting
                eps, momentum, affine, track_running_stats = (
                    module.eps,
                    module.momentum,
                    module.affine,
                    module.track_running_stats,
                )
                in_channels = get_in_channels(pruned_module, name)
                pruned_channels = get_pruned_channels(pruned_module, name)
                # 이전 layer에서 pruning된 channel(in_channels) 처리
                pruned_weight = module.weight.cpu().numpy()
                for t, channel in enumerate(pruned_channels):
                    pruned_weight = np.delete(pruned_weight, channel - t, axis=0)
                pruned_weight = torch.from_numpy(pruned_weight)
                module.weight = torch.nn.Parameter(pruned_weight, requires_grad=True)
                new_batchnorm = nn.BatchNorm2d(
                    in_channels, eps, momentum, affine, track_running_stats
                )  # num_features
                # set_module(model, name, new_batchnorm)
                setattr(prev_module, module_name, new_batchnorm)
            elif isinstance(module, nn.Linear):
                out_features = module.out_features
                bias = False if module.bias == None else True
                in_channels = pruned_module.conv_n_in_channels[1]
                new_linear = nn.Linear(
                    in_features=in_channels, out_features=out_features, bias=bias
                )  # in_features
                setattr(prev_module, module_name, new_linear)
    return model


@torch.no_grad()
def model_structured_prune_for_vgg16(
    model: nn.Module,  # input_size [224,224]
    conv_prun_norm: int = 2,
    conv_prun_rate: float = 0.3,
):

    pruned_module = PrunedModule_vgg()
    linear_flag = True
    # PRUNING FILTERS FOR EFFICIENT CONVNETS
    # https://arxiv.org/pdf/1608.08710.pdf

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # setting
            out_channels, kernel_size, stride, padding = (
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
            )
            bias = False if module.bias is None else True

            # 이전 layer에서 pruning된 channel(in_channels) 처리
            pruned_conv = nn.Conv2d(
                in_channels=pruned_module.in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
            pruned_weight = module.weight.cpu().numpy()
            for t, channel in enumerate(pruned_module.pruned_channels):
                pruned_weight = np.delete(pruned_weight, channel - t, axis=1)
            pruned_weight = torch.from_numpy(pruned_weight)
            pruned_conv.weight = torch.nn.Parameter(pruned_weight, requires_grad=True)

            # pruning n 개 prune (값이 0으로 변경)
            pruning_container = prune.LnStructured.apply(
                pruned_conv,
                name="weight",
                amount=conv_prun_rate,
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
            new_conv = nn.Conv2d(
                in_channels=pruned_module.in_channels,
                out_channels=new_out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
            )
            new_conv.weight = torch.nn.Parameter(new_weight, requires_grad=True)
            set_module(model, name, new_conv)

            pruned_module.in_channels = new_out_channels  # 변수명 생각해보자!!!!!!!!!!!!!!!
            pruned_module.pruned_channels = pruned_channels

        elif isinstance(module, nn.BatchNorm2d):
            eps, momentum, affine, track_running_stats = (
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
            )
            new_batchnorm = nn.BatchNorm2d(
                pruned_module.in_channels, eps, momentum, affine, track_running_stats
            )  # num_features
            set_module(model, name, new_batchnorm)

        elif isinstance(module, nn.Linear) and linear_flag:
            out_features = module.out_features
            bias = False if module.bias == None else True
            new_linear = nn.Linear(
                in_features=pruned_module.in_channels * 49,
                out_features=out_features,
                bias=bias,
            )  # in_features
            set_module(model, name, new_linear)
            linear_flag = False

    return model
