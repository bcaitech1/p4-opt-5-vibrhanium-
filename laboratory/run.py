import os
import os.path as p
import argparse

import torch

import utils.model as M
from .utils.common import read_yaml, calc_macs
from .utils.train_utils import get_dataloader, F1CELoss
from .utils.trainer import TorchTrainer


def run(args, device):
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

    criterion = F1CELoss()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20)
    macs = calc_macs(model, (img_size, img_size))

    train_loader, valid_loader = get_dataloader(
        data_path=data_config["DATA_PATH"],
        batch_size=data_config["BATCH_SIZE"],
        img_size=img_size
    )
    
    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        macs=macs,
        scaler=None,
        device=device,
        model_path=save_path,
        verbose=1
    )
    best_f1, best_acc = trainer.train(
        train_dataloader=train_loader,
        val_dataloader=valid_loader,
        n_epoch=150
    )

    model.load_state_dict(torch.load(save_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model, test_dataloader=valid_loader
    )

    print(f"BEST F1: {best_f1} | BEST ACC: {best_acc}")
    print(f"TEST F1: {test_f1} | TEST ACC: {test_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run")
    parser.add_argument("--config", required=True, type=str, help="config file path")
    parser.add_argument("--output_path", required=True, type=str, help="path to save")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(args, device)