import re
import numpy as np
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.nn as nn
import torch

class PrunedModule(nn.Module):
    def __init__(self, in_channels: int=3, pruned_channels: list=[]):
        self.in_channels = in_channels
        self.pruned_channels = pruned_channels
        
def set_module(model: nn.Module, name: str, new_module):
    p = re.compile('[0-9]')
    layers_name, module_name = name.split('.')[:-1], name.split('.')[-1]
    prev_module = model
    for layer_name in layers_name:
        if p.match(layer_name):
            prev_module = prev_module[int(layer_name)]
        else:
            prev_module = getattr(prev_module, layer_name)
    setattr(prev_module, module_name, new_module)
    
@torch.no_grad()
def model_prune(
    model: nn.Module, # input_size [224,224]
    conv_prun_norm: int = 2,
    conv_prun_rate: float = 0.3,
    ):
    
    pruned_module = PrunedModule()
    linear_flag = True
    # PRUNING FILTERS FOR EFFICIENT CONVNETS
    # https://arxiv.org/pdf/1608.08710.pdf
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # setting
            out_channels, kernel_size, stride, padding = module.out_channels, module.kernel_size, module.stride, module.padding
            bias = False if module.bias is None else True
            # 이전 layer에서 pruning된 channel(in_channels) 처리
            pruned_conv = nn.Conv2d(in_channels=pruned_module.in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            pruned_weight = module.weight.cpu().numpy()
            for t, channel in enumerate(pruned_channels):
                pruned_weight = np.delete(pruned_weight, channel - t, axis=1)
#             pruned_weight = pruned_weight[:, np.array(list(set(range(pruned_weight.shape[1]))-set(pruned_module.pruned_channels))), :, :]
            pruned_weight = torch.from_numpy(pruned_weight)
            pruned_conv.weight = torch.nn.Parameter(pruned_weight,requires_grad=True)
            # pruning n 개 prune (값이 0으로 변경)
            pruning_container = prune.LnStructured.apply(pruned_conv, name='weight', amount=conv_prun_rate, n=conv_prun_norm, dim=0)
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
            new_conv = nn.Conv2d(in_channels=pruned_module.in_channels, out_channels=new_out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
            new_conv.weight = torch.nn.Parameter(new_weight,requires_grad=True)
            set_module(model, name, new_conv)
            pruned_module.in_channels = new_out_channels # 변수명 생각해보자!!!!!!!!!!!!!!!
            pruned_module.pruned_channels = pruned_channels
            
        elif isinstance(module, nn.BatchNorm2d):
            eps, momentum, affine, track_running_stats = module.eps, module.momentum, module.affine, module.track_running_stats
            new_batchnorm = nn.BatchNorm2d(pruned_module.in_channels, eps, momentum, affine, track_running_stats)  # num_features
            set_module(model, name, new_batchnorm)
            
        elif isinstance(module, nn.Linear) and linear_flag :
            out_features = module.out_features
            bias = False if module.bias==None else True
            new_linear = nn.Linear(in_features=pruned_module.in_channels*49, out_features=out_features, bias=bias)  # in_features
            set_module(model, name, new_linear)
            linear_flag = False
            
    return model
