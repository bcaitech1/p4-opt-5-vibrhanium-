from ptflops import get_model_complexity_info
import torch
import torch.nn as nn


def calc_macs(model, input_shape):
    macs, params = get_model_complexity_info(
        model=model,
        input_res=input_shape,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
        ignore_modules=[nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6],
    )
    return macs
