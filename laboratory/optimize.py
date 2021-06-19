import os
import os.path as p
import time
import copy
import argparse

import torch

import utils.model as M
from utils.evaluation import evaluate
from utils.common import read_yaml, check_spec, print_spec
from utils.train_utils import get_dataloader, F1CELoss
from utils.trainer import TorchTrainer
from optimization.decompose import decompose
from optimization.prune import (
    model_structured_prune_for_vgg16, 
    model_structured_prune_for_resnet34, 
    model_structured_prune_for_shufflenet_v2
)


def check_before(model, dataloader, img_size, device):
    saved_model = copy.deepcopy(model)

    before_macs, before_num_parameters = check_spec(model, img_size)
    before_f1, before_accuracy, before_consumed_time = evaluate(model=model, dataloader=dataloader, device=device)

    return [before_consumed_time, before_macs, before_num_parameters, before_f1, before_accuracy]


def optimize_step(config, optim_func, target_model, per_epochs, scaler, device, model_path, *args):
    # -- Do optimization
    optimized_model = optim_func(target_model).to(device)
    print("Success to optimize model")
    macs, num_parameters = check_spec(optimized_model, config["IMG_SIZE"])

    optimizer = torch.optim.AdamW(optimized_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20)
    criterion = F1CELoss()

    train_loader, valid_loader = get_dataloader(
        data_path=config["DATA_PATH"],
        batch_size=config["BATCH_SIZE"],
        img_size=config["IMG_SIZE"]
    )

    trainer = TorchTrainer(
        model=optimized_model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        macs=macs,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1
    )

    start_time = time.time()
    best_f1, best_acc = trainer.train(
        train_dataloader=train_loader,
        val_dataloader=valid_loader,
        n_epoch=per_epochs
    )
    consumed_time = time.time() - start_time
    
    return consumed_time, macs, num_parameters, best_f1, best_acc


def optimize(args, iter_num, per_epochs, device):
    data_config = read_yaml(cfg=args.config)
    alias = data_config["ALIAS"]
    img_size = data_config["IMG_SIZE"]

    save_dir = p.join(args.output_path, alias)
    os.makedirs(save_dir, exist_ok=True)
    save_path = p.join(save_dir, f"{alias}_{img_size}.pt")
    
    print(f"Input image size: {img_size}")
    print(f"Path to save new model: {save_path}")

    model = getattr(
        M, data_config["MODEL"]
    )(num_classes=data_config["NUM_CLASSES"], weight_path=data_config["WEIGHT_PATH"]).to(device)
    print(f"Success to load saved model.")

    _, valid_loader = get_dataloader(
        data_path=data_config["DATA_PATH"],
        batch_size=data_config["BATCH_SIZE"],
        img_size=img_size 
    )

    before_spec = check_before(model, valid_loader, img_size, device)
    print("기존 수치")
    print_spec(*before_spec)
    print("=" * 10)
    print()

    if args.opt_type == 0:
        optim_func = decompose
    elif args.opt_type == 1:
        if data_config["MODEL"] == "VGGNet16":
            optim_func = model_structured_prune_for_vgg16
        if data_config["MODEL"] == "ResNet34":
            optim_func = model_structured_prune_for_resnet34
        if data_config["MODEL"] == "ShuffleNet_v2_x0_5":
            optim_func = model_structured_prune_for_shufflenet_v2
    else:
        print("This type is not supported yet.")
        assert 0

    specs = []
    specs.append(before_spec)
    for i in range(iter_num):
        after_spec = optimize_step(
            config=data_config, 
            optim_func=optim_func, 
            target_model=model, 
            per_epochs=per_epochs, 
            scaler=None, 
            device=device, 
            model_path=f"{save_path}_{i}_step"
        )

        specs.append(after_spec)
        print(f"{i + 1}번째 수치")
        print_spec(*after_spec)
        print("=" * 10)
        print()

    return specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run")
    parser.add_argument("--config", required=True, type=str, help="config file path")
    parser.add_argument("--output_path", required=True, type=str, help="path to save")
    parser.add_argument("--opt_type", required=True, type=int, help="0: decompose, 1: prune, 2: both")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = optimize(args=args, iter_num=3, per_epochs=2, device=device)

    for i, spec in enumerate(specs):
        print(f"{i}-th values")
        print_spec(*spec)
        print("=" * 10)
        print()