import torch
import torch.nn as nn
import torchvision


class VGGNet16(nn.Module):
    def __init__(self, num_classes, weight_path):
        super().__init__()
        self.model = torchvision.models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)

        if weight_path:
            self.model.load_state_dict(torch.load(weight_path))

    def forward(self, x):
        return self.model(x)


class ResNet34(nn.Module):
    def __init__(self, num_classes, weight_path):
        super().__init__()
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)

        if weight_path:
            self.model.load_state_dict(torch.load(weight_path))

    def forward(self, x):
        return self.model(x)


class ShuffleNet_v2_x0_5(nn.Module):
    def __init__(self, num_classes, weight_path):
        super().__init__()
        self.model = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
        self.model.fc = nn.Linear(1024, num_classes)

        if weight_path:
            self.load_state_dict(torch.load(weight_path))

    def forward(self, x):
        return self.model(x)