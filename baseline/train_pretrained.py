"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
from datetime import datetime
import os
import yaml
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F

from src.dataloader import create_dataloader
from src.loss import CustomCriterion

from src.trainer import TorchTrainer

from src.utils.common import get_label_counts, read_yaml
from src.utils.macs import calc_macs
from src.utils.prune import (
    model_structured_prune_for_resnet34,
    model_structured_prune_for_shufflenet_v2,
)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )


def train_pretrained(
    model_name: str,
    data_config: Dict[str, Any],
    log_dir: str,
    log_name: str,
    fp16: bool,
    device: torch.device,
    num_classes: int = 9,
) -> Tuple[float, float, float]:
    """Train."""

    img_size = data_config["IMG_SIZE"]

    # model = getattr(__import__("src.models", fromlist=[""]), model_name)()  # "Resnet34"
    model = torchvision.models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, num_classes)

    if args.prune_resnet34:
        print("Start pruning")
        MODEL_PATH = f"/opt/ml/input/exp/Resnet34/Resnet34_{data_name}.pt"
        model = model_structured_prune_for_resnet34(model)
        print("End pruning")
        model.load_state_dict(torch.load(MODEL_PATH))

    model.to(device)
    model_path = os.path.join(log_dir, f"{log_name}.pt")
    print(f"Model save path: {model_path}")

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Calc macs
    macs = calc_macs(model, (3, img_size, img_size))
    print(f"macs: {macs}")

    # Create optimizer, scheduler, criterion
    optimizer = torch.optim.SGD(
        model.parameters(), lr=data_config["INIT_LR"], momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=20
    )
    # criterion = FocalLoss()
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
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        macs=macs,
        scaler=scaler,
        device=device,
        model_path=model_path,
        wandb_name=log_name,  # log_name,
    )
    best_lbs, best_test_acc, best_test_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )
    print(
        "best_lbs",
        best_lbs,
        "best_test_acc:",
        best_test_acc,
        "best_test_f1:",
        best_test_f1,
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
        "--data",
        default="/opt/ml/p4-opt-5-vibrhanium-/baseline/configs/data/taco_sample.yaml",
        type=str,
        help="data config",
    )
    parser.add_argument(
        "--model", required=True, type=str, help="model name",
    )
    parser.add_argument(
        "--prune_resnet34", default=False, type=str, help="pruning resnet34",
    )
    parser.add_argument(
        "--num_classes", required=True, type=int, help="number of classes",
    )
    args = parser.parse_args()

    import sys
    import os.path as p

    ROOT_PATH = "/opt/ml/"
    BASE_PATH = p.join(ROOT_PATH, "p4-opt-5-vibrhanium-/baseline")
    sys.path.append(BASE_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cur_time = datetime.now().strftime("%m%d_%H%M")
    print("cur_time", cur_time)

    data_config = read_yaml(cfg=args.data)
    model_name = args.model
    data_name = args.data.split("/")[-1][:-5]

    log_dir = os.path.join("/opt/ml/input/exp", model_name)
    os.makedirs(log_dir, exist_ok=True)
    log_name = f"{model_name}_{data_name}_{data_config['EPOCHS']}_{cur_time}"
    print("log_name", log_name)

    test_loss, test_f1, test_acc = train_pretrained(
        model_name=model_name,
        data_config=data_config,
        log_dir=log_dir,
        log_name=log_name,
        fp16=False,
        device=device,
        num_classes=args.num_classes,
    )
