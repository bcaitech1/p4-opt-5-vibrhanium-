import os
import re
import sys
import yaml
import logging
import argparse
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import wandb
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from dotenv import load_dotenv

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.macs import calc_macs


def suggest_from_config(trial, config_dict, key, name=None):
    """sugget value from config

    Args:
        trial (optuna.trial.Trial): optuna trial
        config_dict (Dict[str, Any]): config dict from yaml file
        key (str): key in config dict
        name (str, optional): name argument in suggest function if name is None use key argument. Defaults to None.

    Raises:
        ValueError: raise when 'type' value of config_dict[key] is not 'categorical', 'float', or 'int'

    Returns:
        Any: suggested value
    """    
    par = config_dict[key]
    if par['type'] == 'categorical':
        return trial.suggest_categorical (
                                  name    = name if name else key,
                                  choices = list ( par [ 'choices' ] ) ,
                                )
    elif par['type'] == 'float':
        return trial.suggest_float (
                            name = name if name else key,
                            low  = float ( par [ 'low'  ] ) ,
                            high = float ( par [ 'high' ] ) ,
                            step = float ( par [ 'step' ] ) if par.get('step') else None  ,
                            log  = bool  ( par [ 'log'  ] ) if par.get('log') else False ,
                          )
    elif par['type'] == 'int':
        return trial.suggest_int (
                          name = name if name else key,
                          low  = float ( par [ 'low'  ] ) ,
                          high = float ( par [ 'high' ] ) ,
                          step = float ( par [ 'step' ] ) if par.get('step') else 1     ,
                          log  = bool  ( par [ 'log'  ] ) if par.get('log') else False ,
                        )
    else:
        raise ValueError ('Trial suggestion not implemented.')


def search_model(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search model structure from user-specified search space.
    Returns:
        Dict[str, Any]: Sampled model architecture config.
    """
    # 초기 설정 setting
    num_cells = base_config['num_cells']
    normal_cells = module_config['normal_cells']
    reduction_cells = module_config['reduction_cells']
    model_config = {
        'input_channel': 3,
        'depth_multiple': 1.0,
        'width_multiple': 1.0,
        'backbone': [],
        }

    dropout_rate = suggest_from_config(trial, base_config, "dropout_rate")
    
    # Sample Normal Cell(NC)
    n_nc = num_cells['value'] 

    low, high = num_cells['low'], num_cells['high']
    # deeper layer, more features
    nx = [trial.suggest_int(name=f"n{i}_repeat", low=low*(i), high=high*(i)) for i in range(1,n_nc+1)] 
    
    ncx = [[] for _ in range(n_nc)]
    ncx_args = [[] for _ in range(n_nc)]
    for i in range(n_nc):
        nc = suggest_from_config(trial, base_config, 'normal_cells', f'normal_cells_{i}')
        nc_config = normal_cells[nc]

        nc_args = []
        for arg, value in nc_config.items(): #out_channel, {name:oc, type:int ... } | kernel_size, {asd}
            if isinstance(value, dict): 
                nc_args.append(suggest_from_config(trial, nc_config, arg, f'normal_cells_{i}/{arg}'))
            else:
                nc_args.append(value)
        ncx[i] = nc
        ncx_args[i] = nc_args

    # Sample Reduction Cell(RC)
    n_rc = n_nc-1
    rcx = [[] for _ in range(n_rc)]
    rcx_args = [[] for _ in range(n_rc)]
    for i in range(n_rc):
        rc = suggest_from_config(trial, base_config, 'reduction_cells', f'reduction_cells_{i}')
        rc_config = reduction_cells[rc]

        rc_args = []
        for arg, value in rc_config.items():
            if isinstance(value, dict):
                rc_args.append(suggest_from_config(trial, rc_config, arg, f'reduction_cells_{i}/{arg}'))
            else:
                rc_args.append(value)
        rcx[i] = rc
        rcx_args[i] = rc_args

    for i in range(n_rc):
        model_config["backbone"].append([nx[i], ncx[i], ncx_args[i]])
        model_config["backbone"].append([1,     rcx[i], rcx_args[i]])
    model_config["backbone"].append([nx[-1], ncx[-1], ncx_args[-1]])
    model_config["backbone"].append([1, "GlobalAvgPool", []])
    model_config["backbone"].append([1, "Flatten", []])
    model_config["backbone"].append([1, "Dropout", [dropout_rate]])
    model_config["backbone"].append([1, "Linear", [num_class]])

    # save model dict to .yaml file
    save_config_dir = os.path.join(save_config_dir_base, str(trial.number))
    os.makedirs(save_config_dir, exist_ok=True)
    config_fn = f"{save_config_fn_base}_{trial.number}_model.yaml"
    with open(os.path.join(save_config_dir, config_fn), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    return model_config


def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search hyperparameter from user-specified search space.
    Returns:
        Dict[str, Any]: Sampled hyperparam configs.
    """
    # search hyperparameter
    epochs = suggest_from_config(trial, base_config, 'epochs')
    batch_size = suggest_from_config(trial, base_config, 'batch_size')
    img_size = suggest_from_config(trial, base_config, 'img_size')
    hyperparam_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size
        }
    
    # optimizer
    optimizer_name = suggest_from_config(trial, base_config, "optimizer")  # AdamW
    optimizer_params = optimizer_config[optimizer_name].keys()  #lr, beta1, beta2s
    params_dict = {}
    for param in optimizer_params: # lr, betas
        if param == "betas":
            beta1 = suggest_from_config(
                trial, optimizer_config[optimizer_name], param, f"optimizer/{param}"
            )  
            beta2 = optimizer_config[optimizer_name][param]["beta2"]
            params_dict["betas"] = (beta1, beta2)
            continue

        params_dict[param] = suggest_from_config(
            trial, optimizer_config[optimizer_name], param, f"optimizer/{param}"
        )
    hyperparam_config["optimizer"] = optimizer_name
    hyperparam_config["optimizer_params"] = params_dict
    
    # scheduler
    scheduler_name = suggest_from_config(trial, base_config, "scheduler")  # CosineAnnealingLR
    scheduler_params = scheduler_config[scheduler_name].keys()  # T_max eta_min last_epoch
    params_dict = {}
    for param in scheduler_params:
        params_dict[param] = suggest_from_config(trial, scheduler_config[scheduler_name], param, f"scheduler/{param}")
    hyperparam_config["scheduler"] = scheduler_name
    hyperparam_config["scheduler_params"] = params_dict
    
    # criterion
    criterion_name = suggest_from_config(trial, base_config, "criterion") 
    hyperparam_config["criterion"] = criterion_name

    # save hyper-parameter dict to .yaml file (dict -> yaml)
    save_config_dir = os.path.join(save_config_dir_base, str(trial.number))
    config_fn = f"{save_config_fn_base}_{trial.number}_hyperparameter.yaml"
    with open(os.path.join(save_config_dir, config_fn), "w") as f:
        yaml.dump(hyperparam_config, f, default_flow_style=False)

    return hyperparam_config


def train_model(trial,
    model_instance: nn.Module,
    hyperparam_config: Dict[str, Any]
    ) -> nn.Module:
    """Create trainer and train.
    Args:
        model_config: Torch model to be trained.
        hyperparams: Hyperparams for training
            (e.g. optimizer, lr, batch_size, ...)
    Returns:
        nn.Module: Trained torch model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    fp16 = data_config["FP16"] #floating point 16
    
    # search hyperparameter
    epochs = hyperparam_config["epochs"]
    batch_size = hyperparam_config["batch_size"]
    
    # optimizer
    optimizer_name = hyperparam_config["optimizer"]
    optimizer_params = hyperparam_config["optimizer_params"]
    optimizer = getattr(optim, optimizer_name)(
        model_instance.parameters(), **optimizer_params
    )
    
    # scheduler
    scheduler_name = hyperparam_config["scheduler"]
    scheduler_params = hyperparam_config["scheduler_params"]
    scheduler = getattr(lr_scheduler, scheduler_name)(optimizer=optimizer, **scheduler_params) # dictionary unpacking
    
    # criterion
    criterion_name = hyperparam_config["criterion"] 
    criterion = getattr(nn, criterion_name)()

    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )
    
    data_config['IMG_SIZE'] = hyperparam_config["img_size"]
    data_config['BATCH_SIZE'] = hyperparam_config["batch_size"]
    train_dl, val_dl, test_dl = create_dataloader(data_config)
    
    if args.save_model:
        save_model_dir = os.path.join(save_model_dir_base, str(trial.number))
        os.makedirs(save_model_dir, exist_ok=True)
        model_fn = "best.pt"
        model_path = os.path.join(save_model_dir, model_fn)
    else:
        model_path = None
    # Create trainer
    trainer = TorchTrainer(
        model=model_instance.model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=epochs,
        val_dataloader=val_dl if val_dl else test_dl,
    )
    return best_f1


def objective(trial: optuna.trial.Trial) -> float:
    """Get objective score.
    
    Args:
        trial: optuna trial object
    Returns:
        float: Score value.
            whether to maximize, minimize will determined in optuna study setup.
    """
    model_config = search_model(trial)
    hyperparam_config = search_hyperparam(trial)

    model_instance = Model(model_config, verbose=True)
    model_instance.model.to(device)

    # Calc macs
    macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
    print(f"macs: {macs}")

    best_f1 = train_model(trial, model_instance, hyperparam_config)
    run = wandb.init(project='OPT', name = f'{cur_time}_{trial.number}' , reinit=False)
    wandb.log({'f1':best_f1, 'MACs':macs})
    run.finish()
    
    return best_f1, macs


if __name__ == "__main__":
    # Setting base
    cur_time = datetime.now().strftime("%m%d_%H%M")
    print("cur_time", cur_time)
    parser = argparse.ArgumentParser(description="Train model using optuna.")
    parser.add_argument(
        "--n_trials", default=10, type=int, help="optuna optimize n_trials"
    )
    parser.add_argument(
        "--study_name", default=None, type=str, help="optuna study alias name"
    )
    parser.add_argument(
        "--save_model", default=False, type=bool, help="choose save model or not save"
    )
    parser.add_argument(
        "--base", default="configs/optuna_config/base_config.yaml", type=str, help="base config"
    )
    parser.add_argument(
        "--module", default="configs/optuna_config/module_config.yaml", type=str, help="module config"
    )
    parser.add_argument(
        "--optimizer", default="configs/optuna_config/optimizer_config.yaml", type=str, help="optimizer config"
    )
    parser.add_argument(
        "--scheduler", default="configs/optuna_config/scheduler_config.yaml", type=str, help="scheduler config"
    )
    parser.add_argument(
        "--data", default="configs/data/taco_sample.yaml", type=str, help="data config"
    )

    args = parser.parse_args()

    # Setting directory - for save best trials model weight
    if args.save_model:
        save_model_dir_base = f"/opt/ml/input/optuna_exp/{cur_time}"
        save_model_fn_base = cur_time

    # Setting directory - for save [best/all] trials model config
    save_config_dir_base = f"/opt/ml/input/config/optuna_model/{cur_time}"
    save_config_fn_base = cur_time

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_class = 9

    # Create config
    data_config = read_yaml(args.data)
    base_config = read_yaml(args.base)
    module_config = read_yaml(args.module) 
    optimizer_config = read_yaml(args.optimizer) 
    scheduler_config = read_yaml(args.scheduler) 
    
    # DB setup
    if args.study_name:
        load_dotenv(verbose=True)
        DB_HOST = os.getenv("DB_HOST")
        DB_NAME = os.getenv("DB_NAME")
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASSWORD")

        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        storage_name = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    
    # Optuna study
    study = optuna.create_study(storage=storage_name,
                                study_name=args.study_name if args.study_name else None,
                                load_if_exists=True if args.study_name else False,
                                directions=["maximize", "minimize"])

    study.optimize(objective, n_trials=args.n_trials)
    
    # # Setting directory - for visualization
    # visualization_dir = "/opt/ml/output/visualization_result"
    # os.makedirs(visualization_dir, exist_ok=True)

    # # Visualization
    # fig = optuna.visualization.plot_pareto_front(study)
    # fig.write_html(os.path.join(visualization_dir, f"{save_config_fn_base}_pareto_front.html"))
