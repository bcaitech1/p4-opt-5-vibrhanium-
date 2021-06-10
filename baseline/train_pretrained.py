"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
from datetime import datetime
import os
import yaml
from typing import Any, Dict, Tuple, Union
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.trainer_pretrained_wandb import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.macs import calc_macs
from src.utils.torch_utils import check_runtime, model_info

import timm

# model_list = [
#     'alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
#     'vgg16', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34',
#     'resnet50', 'resnet101', 'resnet152', 'squeezenet1_0', 'squeezenet1_1',
#     'densenet121', 'densenet169', 'densenet161', 'densenet201', 'inception_v3',
#     'googlenet', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5',
#     'shufflenet_v2_x2_0', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
#     'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
#     'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3'
# ]

model_name = "shufflenet_v2_x0_5"
Model = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
Model = Model.fc.out_features = 9

# timm사용시에는 이렇게
# Model = timm.create_model("efficientnet_b0", num_classes=9, pretrained=True)


def train_pretrained(
    model_name: str,
    from_pretrained: str,
    log_name: str,
    model_config: None,
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model = Model

    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Calc macs
    macs = calc_macs(model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
    print(f"macs: {macs}")

    # Create optimizer, scheduler, criterion
    optimizer = torch.optim.SGD(
        model.parameters(), lr=data_config["INIT_LR"], momentum=0.9
    )
    # adamp.AdamP(model.parameters(), lr=data_config["INIT_LR"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=len(train_dl),
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model_name=model_name,
        model=model,
        model_macs=macs,
        log_name=log_name,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")

    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    args = parser.parse_args()

    model_config = None

    data_config = read_yaml(cfg=args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_name = f"{model_name}_{now_str}"
        log_dir = os.path.join(
            "/opt/ml/input/pretrained_exp", f"{model_name}_{now_str}"
        )
        os.makedirs(log_dir, exist_ok=True)

        test_loss, test_f1, test_acc = train_pretrained(
            model_name=model_name,
            from_pretrained="True",
            log_name=log_name,
            model_config=model_config,
            data_config=data_config,
            log_dir=log_dir,
            fp16=data_config["FP16"],
            device=device,
        )
    except NotImplementedError as e:
        print(model_name)
        print(e)
    except:
        print(f"cant do with model {model_name}")
