import os

import torch
import torch.nn as nn
import torchvision

MODEL_PATH_BASE = "/opt/ml/p4-opt-5-vibrhanium-/laboratory/output/res_224_pruning/res_224_pruning_0_step"
MODEL_ARCITECTURE_PATH = f"{MODEL_PATH_BASE}_architecture.pt"
MODEL_WEIGHT_PATH = f"{MODEL_PATH_BASE}_weight.pt"

SCRIPT_MODEL_FILE_NAME = f"torch_script_{os.path.basename(MODEL_WEIGHT_PATH)}"
SCRIPT_MODEL_PATH = os.path.join(
    "/opt/ml/input/exp_torch_script", SCRIPT_MODEL_FILE_NAME
)
print(f"saved at {SCRIPT_MODEL_PATH}")

model = torch.load(MODEL_ARCITECTURE_PATH)
model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
model.to("cpu")
model.eval()

input = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, input)
traced_script_module.save(SCRIPT_MODEL_PATH)
