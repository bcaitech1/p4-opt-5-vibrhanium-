import torch
import torch.nn as nn
import torchvision.models


class Resnet34(nn.Module):
    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def load_weight(self, weight):
        self.model.load_state_dict(torch.load(weight))
