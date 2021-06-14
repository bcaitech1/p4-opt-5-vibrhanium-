import torch
import torch.nn as nn
import torchvision.models


class Shufflenet_v205(nn.Module):
    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.model = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
        self.model.fc = torch.nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.model(x)
