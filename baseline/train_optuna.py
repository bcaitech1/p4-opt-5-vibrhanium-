import os
import yaml
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

import optuna

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.macs import calc_macs


def suggest_from_config(trial, config_dict, name, idx=''):
    par = config_dict[name]
    if par['type'] == 'categorical':
        return trial.suggest_categorical (
                                  name    = str  ( par [ 'name'    ] ) + str(idx) ,
                                  choices = list ( par [ 'choices' ] ) ,
                                )
    elif par['type'] == 'float':
        return trial.suggest_float (
                            name = str   ( par [ 'name' ] ) + str(idx) ,
                            low  = float ( par [ 'low'  ] ) ,
                            high = float ( par [ 'high' ] ) ,
                            step = float ( par [ 'step' ] ) if par.get('step') else None  ,
                            log  = bool  ( par [ 'log'  ] ) if par.get('log') else False ,
                          )
    elif par['type'] == 'int':
        return trial.suggest_int (
                          name = str   ( par [ 'name' ] ) + str(idx) ,
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
    
    # Sample Normal Cell(NC)
    num_cells = base_config['num_cells']
    n_nc = num_cells['value'] 

    low, high = num_cells['low'], num_cells['high']
    nx = [trial.suggest_int(name=f"n{i}_repeat", low=low*(i), high=high*(i)) for i in range(1,n_nc+1)]
    
    ncx = [[] for _ in range(n_nc)]
    ncx_args = [[] for _ in range(n_nc)]
    for i in range(n_nc):
        nc = suggest_from_config(trial, base_config, 'normal_cells', idx=i)
        nc_config = module_config['normal_cells'][nc]

        nc_args = []
        for arg, value in nc_config.items():
            if isinstance(value, dict):
                nc_args.append(suggest_from_config(trial, nc_config, arg, idx=i))
            else:
                nc_args.append(value)
        ncx[i] = nc
        ncx_args[i] = nc_args

    # Sample Reduction Cell(RC)
    n_rc = n_nc-1
    rcx = [[] for _ in range(n_rc)]
    rcx_args = [[] for _ in range(n_rc)]
    for i in range(n_rc):
        rc = suggest_from_config(trial, base_config, 'reduction_cells', idx=i)
        rc_config = module_config['reduction_cells'][rc]

        rc_args = []
        for arg, value in rc_config.items():
            if isinstance(value, dict):
                rc_args.append(suggest_from_config(trial, rc_config, arg, idx=i))
            else:
                rc_args.append(value)
        rcx[i] = rc
        rcx_args[i] = rc_args
        
    model_config = {
        "input_channel": 3,
        "depth_multiple": 1.0,
        "width_multiple": 1.0,
        "backbone": []
        }


    model_config["backbone"] = []
    for i in range(n_rc):
        model_config["backbone"].append([nx[i], ncx[i], ncx_args[i]])
        model_config["backbone"].append([1,     rcx[i], rcx_args[i]])
    model_config["backbone"].append([nx[-1], ncx[-1], ncx_args[-1]])
    model_config["backbone"].append([1, "GlobalAvgPool", []])
    model_config["backbone"].append([1, "Flatten", []])
    model_config["backbone"].append([1, "Linear", [10]])

    # save model dict to .yaml file
    config_fn = f"{save_config_fn_base}_{trial.number}.yaml"
    with open(os.path.join(save_config_dir, config_fn), "w") as f:
        yaml.dump(save_config_dir, f, default_flow_style=False)
    
    return model_config


def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search hyperparameter from user-specified search space.
    Returns:
        Dict[str, Any]: Sampled hyperparam configs.
    """
    epochs = suggest_from_config(trial, base_config, 'epochs')
    batch_size = suggest_from_config(trial, base_config, 'batch_size')
    max_lr = suggest_from_config(trial, base_config, 'max_learning_rate')

    # Sample optimizer
    optimizer = suggest_from_config(trial, base_config, 'optimizer')

    optimizer_args = {}
    for args, value in optimizer_config[optimizer].items():
        optimizer_args[args] = suggest_from_config(trial, optimizer_config[optimizer], args)
        if args == 'beta2':
            optimizer_args['betas'] = (optimizer_args['beta1'], optimizer_args['beta2'])
            del optimizer_args['beta1']
            del optimizer_args['beta2']

    
    img_size = suggest_from_config(trial, base_config, 'img_size')
    data_config['IMG_SIZE'] = img_size

    hyperparam_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "max_lr": max_lr,
        "optimizer": optimizer,
        "optimizer_args": optimizer_args,
        # "lr": lr,
        "img_size": img_size
    }

    # save hyper-parameter dict to .yaml file (dict -> yaml)
    hyperpraam_config_fn = f"{save_config_fn_base}_{trial.number}_hyperparameter.yaml"
    with open(os.path.join(save_config_dir, hyperpraam_config_fn), "w") as f:
        yaml.dump(hyperparam_config, f, default_flow_style=False)

    return hyperparam_config


def train_model(
    model_instance: nn.Module,
    hyperparams: Dict[str, Any]
    ) -> nn.Module:
    """Create trainer and train.
    Args:
        save_config_dir: Torch model to be trained.
        hyperparams: Hyperparams for training
            (e.g. optimizer, lr, batch_size, ...)
    Returns:
        nn.Module: Trained torch model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    fp16 = data_config["FP16"]
    
    # Create optimizer, scheduler, criterion
    optimizer = getattr(optim, hyperparams["optimizer"])(
        model_instance.model.parameters(), **hyperparams["optimizer_args"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=hyperparams["max_lr"],
        steps_per_epoch=len(train_dl),
        epochs=hyperparams["epochs"],
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
    global global_best_f1
    trainer = TorchTrainer(
        model=model_instance.model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=save_model_path,
        verbose=1,
        global_best_f1=global_best_f1
    )
    best_acc, global_best_f1, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=hyperparams["epochs"],
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
    hyperparams = search_hyperparam(trial)

    model_instance = Model(model_config, verbose=True)
    model_instance.model.to(device)

    # Calc macs
    macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
    print(f"macs: {macs}")

    best_f1 = train_model(model_instance, hyperparams)

    return best_f1, macs


if __name__ == '__main__':
    # Setting directory and file name
    cur_time = datetime.now().strftime('%m%d_%H%M')
    save_config_dir = "configs/optuna_model"    # model configs dir
    save_config_fn_base = cur_time              # model configs file name

    save_model_dir = "optuna_exp"               # model weight dir
    save_model_path = os.path.join(save_model_dir, f"{cur_time}_best.pt")  # model weight file path
    global_best_f1 = 0  # 모든 trial 중 best f1, save_model에 사용 (임시)

    save_visualization_dir = 'visualization_result'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Create config file
    data_config = read_yaml('configs/data/taco.yaml')
    base_config = read_yaml('configs/optuna_config/base_config.yaml')
    module_config = read_yaml('configs/optuna_config/module_config.yaml')
    optimizer_config = read_yaml('configs/optuna_config/optimizer_config.yaml')

    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Oputna study
    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=100)

    # Visualization
    fig = optuna.visualization.plot_pareto_front(study)
    fig.write_html(os.path.join(visualization_dir, f"{save_config_fn_base}.html"))
