import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torch.nn as nn
import torch

def model_prune(
    model: nn.Module,
    
    conv_prun_norm: int = 1,
    conv_prun_rate: float = 0.2,
    linear_prun_norm: int = 1,
    linear_prun_rate: float = 0.4,
    structured: bool = False
):
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            if structured:
                prune.ln_structured(module, name='weight', amount=conv_prun_rate, n=conv_prun_norm, dim=0)
            else:
                if conv_prun_norm == 1:
                    prune.l1_unstructured(module, name='weight', amount=conv_prun_rate)
                else:
                    prune.l2_unstructured(module, name='weight', amount=conv_prun_rate)
            prune.remove(module,'weight')

        elif isinstance(module, torch.nn.modules.linear.Linear):
            if structured:
                prune.ln_structured(module, name='weight', amount=linear_prun_rate, n=conv_prun_norm, dim=0)
            else:
                if linear_prun_norm == 1:
                    prune.l1_unstructured(module, name='weight', amount=linear_prun_rate)
                else:
                    prune.l2_unstructured(module, name='weight', amount=linear_prun_rate)
                
            prune.remove(module,'weight')
    
    print("-"*10+'prun 적용 모듈'+"-"*10)
    print(dict(model.named_buffers()).keys())
    
    return model
