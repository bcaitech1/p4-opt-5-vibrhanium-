"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
from datetime import datetime
import os
import yaml
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.macs import calc_macs
from src.utils.torch_utils import check_runtime, model_info


def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, 'data.yml'), 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, 'model.yml'), 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model_instance = Model(model_config, verbose=True)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        model_instance.model.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.model.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Calc macs
    macs = calc_macs(model_instance.model, (3, hyperparam_config["img_size"], hyperparam_config["img_size"]))
    print(f"macs: {macs}")

    # Create optimizer, scheduler, criterion
    # optimizer
    optimizer_name = hyperparam_config["optimizer"]  # AdamW
    optimizer_params = ["lr", "betas"]  ### 이후에 수정 예정
    temp_dict = {}
    for param in optimizer_params: # lr, betas
        if param == "betas":
            temp_dict["betas"] = (hyperparam_config["betas"], 0.9999)  ### 이후에 수정 예정
            continue

        temp_dict[param] = hyperparam_config[param]  ### 이후에 수정 예정

    optimizer = getattr(optim, optimizer_name)(
        model_instance.parameters(), **temp_dict
    )

    # scheduler
    scheduler_name = hyperparam_config["scheduler"]
    scheduler_params = ["T_max", "eta_min", "last_epoch"] ### 이후에 수정 예정
    temp_dict = {}
    for param in scheduler_params:
        temp_dict[param] = hyperparam_config[param]
    
    scheduler = getattr(lr_scheduler, scheduler_name)(optimizer=optimizer, **temp_dict) # dictionary unpacking
    
    # criterion
    criterion_name = hyperparam_config["criterion"]
    criterion = getattr(nn, criterion_name)()

    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=model_instance.model,
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
        n_epoch=hyperparam_config["epochs"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model_instance.model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model_instance.model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    parser.add_argument(
        "--model", required=True, type=str, help="model config path"
    )
    parser.add_argument(
        "--hyperparam", required=True, type=str, help="hyper-parameter config path"
    )
    args = parser.parse_args()

    # Create config
    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)
    hyperparam_config = read_yaml(cfg=args.hyperparam)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join("/opt/ml/input/exp", datetime.now().strftime("%m%d_%H%M"))
    os.makedirs(log_dir, exist_ok=True)

    test_loss, test_f1, test_acc = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )
