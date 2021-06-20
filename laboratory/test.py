import torch
import torch.nn as nn
import torchvision.models

from utils.evaluation import evaluate
from utils.common import check_spec, print_spec
from utils.train_utils import get_dataloader


MODEL = "ResNet34"
NUM_CLASSES = 8
WEIGHT_PATH = "/opt/ml/p4-opt-5-vibrhanium-/laboratory/weights/res.pt"
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_before(model, dataloader, img_size, device):
    before_macs, before_num_parameters = check_spec(model, img_size)
    before_f1, before_accuracy, before_consumed_time = evaluate(
        model=model, dataloader=dataloader, device=device
    )

    return [
        before_consumed_time,
        before_macs,
        before_num_parameters,
        before_f1,
        before_accuracy,
    ]


if MODEL == "ResNet34":
    model = torchvision.models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHT_PATH))
    model.to(device)
elif MODEL == "ShuffleNet_v2_x0_5":
    model = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
    model.fc = torch.nn.Linear(1024, NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHT_PATH))
    model.to(device)
elif MODEL == "VGGNet16":
    model = torchvision.models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHT_PATH))
    model.to(device)


_, valid_loader = get_dataloader(
    data_path="/opt/ml/input/data/", batch_size=64, img_size=IMG_SIZE,
)

spec = check_before(model, valid_loader, IMG_SIZE, device)
print_spec(*spec)
