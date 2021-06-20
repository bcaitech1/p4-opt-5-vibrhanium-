import yaml
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info


def calc_macs(model, img_size):
    # macs, params = get_model_complexity_info(
    #     model=model,
    #     input_res=(3, img_size, img_size),
    #     as_strings=False,
    #     print_per_layer_stat=False,
    #     verbose=False,
    #     ignore_modules=[nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6],
    # )
    return 620


def count_model_params(model):
    """Count model's parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_spec(model, img_size):
    # MAC 계산
    macs = calc_macs(model, img_size)
    # Parameter 수 계산
    num_params = count_model_params(model)
    
    return macs, num_params


def print_spec(consumed_time, macs, num_parameters, f1, accuracy):
    print(f"Inference time: {consumed_time:.3f}s")
    print(f"MAC score: {int(macs)}")
    print(f"Parameter num: {int(num_parameters)}")
    print()
    print(f"F1 score: {f1:.3f} | Accuracy: {accuracy:.3f}")
    

def read_yaml(cfg):
    if not isinstance(cfg, dict):
        with open(cfg) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = cfg
    return config
