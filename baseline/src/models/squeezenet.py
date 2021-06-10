import torch.nn as nn
import torchvision.models


class Squeezenet1_1(nn.Module):
    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x):
        return self.model(x)
