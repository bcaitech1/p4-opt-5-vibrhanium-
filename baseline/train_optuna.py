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



def suggest_from_config(trial, config_dict, name):
    par = config_dict[name]
    if par['type'] == 'categorical':
        return trial.suggest_categorical (
                                  name    = str  ( par [ 'name'    ] ) ,
                                  choices = list ( par [ 'choices' ] ) ,
                                )
    elif par['type'] == 'float':
        return trial.suggest_float (
                            name = str   ( par [ 'name' ] ) ,
                            low  = float ( par [ 'low'  ] ) ,
                            high = float ( par [ 'high' ] ) ,
                            step = float ( par [ 'step' ] ) if par.get('step') else None  ,
                            log  = bool  ( par [ 'log'  ] ) if par.get('log') else False ,
                          )
    elif par['type'] == 'int':
        return trial.suggest_int (
                          name = str   ( par [ 'name' ] ) ,
                          low  = float ( par [ 'low'  ] ) ,
                          high = float ( par [ 'high' ] ) ,
                          step = float ( par [ 'step' ] ) if par.get('step') else 1     ,
                          log  = bool  ( par [ 'log'  ] ) if par.get('log') else False ,
                        )
    else:
        raise ValueError ('Trial suggestion not implemented.')

def search_model(trial):  # optuna.trial.Trial) -> Dict[str, Any]:
    """Search model structure from user-specified search space.
    Returns:
        Dict[str, Any]: Sampled model architecture config.
    """
    
    # Sample Normal Cell(NC)
    num_cells = optuna_config['num_cells']
    n_nc = num_cells['value'] 

    low, high = num_cells['low'], num_cells['high']
    ncx_layer = [trial.suggest_int(name=f"n{i}_repeat", low=low*(i), high=high*(i) for i in range(1,n_nc+1))]
    
    ncx = [[] for _ in range(n_nc)]  # [[], [], []]
    ncx_args = [[] for _ in range(n_nc)]
    
    for i in range(n_nc):
        nc = "Conv"  # suggest_from_config
        nc_config = modules_config['normal_cell'][nc]

        nc_args = []
        for arg, value in nc_config.items():
            if isinstance(value, dict):
                nc_args.append(suggest_from_config(trial, nc_config, arg))
            else:
                nc_args.append(value)
        ncx_args[i] = nc_args
        
    # ncx_args

    # reduction cell
    n_rc = 1
    rcx_args = [[] for _ in range(n_rc)]

    for i in range(n_rc):
        rc = "InvertedResidualv3"  # suggest_from_config
        rc_config = modules_config['reduction_cell'][rc]

        rc_args = []
        for arg, value in rc_config.items():
            if isinstance(value, dict):
                rc_args.append(suggest_from_config(trial, rc_config, arg))
            else:
                rc_args.append(value)
        rcx_args[i] = rc_args
    
    # rcx_args
        
    model_config = {
        "input_channel": 3,
        "depth_multiple": 1.0,
        "width_multiple": 1.0,
        "backbone": []
        }
    
    model_config["backbone"].append([n1, nc1, nc1_args])
    model_config["backbone"].append([1, rc1, rc1_args])
    model_config["backbone"].append([n2, nc2, nc2_args])
    model_config["backbone"].append([1, rc2, rc2_args])
    model_config["backbone"].append([n3, nc3, nc3_args])
    model_config["backbone"].append([1, "GlobalAvgPool", []])
    model_config["backbone"].append([1, "Flatten", []])
    model_config["backbone"].append([1, "Linear", [10]])

    # save model dict to .yaml file (dict -> yaml)
    model_fn = f"{model_fn_base}_{trial.number}.yaml"
    with open(os.path.join(model_dir, model_fn), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    return model_config


def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search hyperparameter from user-specified search space.
    Returns:
        Dict[str, Any]: Sampled hyperparam configs.
    """
    # epochs = trial.suggest_int("epochs", 10, 30, step=10)
    epochs = suggest_from_config(trial, optuna_config, 'epochs')
    batch_size = suggest_from_config(trial, optuna_config, 'batch_size')
    lr = suggest_from_config(trial, optuna_config, 'learning_rate')

    # Sample optimizer
    optimizer = suggest_from_config(trial, optuna_config, 'optimizer')

    # Optimizer args are conditional! ##### 수정 필요
    if optimizer == "Adam":
        # More aggressive lr!
        lr = suggest_from_config(trial, optimizer['Adam'], 'lr') ##### 수정 필요

        # Adam only params
        beta1 = suggest_from_config(trial, optimizer['Adam'], 'beta1')
        beta2 = suggest_from_config(trial, optimizer['Adam'], 'beta2')
    
        optimizer_args = [(beta1, beta2)]

    elif optimizer == "SGD":
        pass
    elif optimizer == "Adagrad":
        pass  
    elif optimizer == "AdamW":
        pass

    hyperparam_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "optimizer": optimizer,
        "optimizer_args": optimizer_args,
    }

    return hyperparam_config


def train_model(
    model_instance: nn.Module,
    hyperparams: Dict[str, Any]
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
    fp16 = data_config["FP16"]
    
    # Create optimizer, scheduler, criterion
    optimizer = getattr(optim, hyperparams["optimizer"])(
        model_instance.model.parameters(), hyperparams["lr"], *hyperparams["optimizer_args"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=hyperparams["lr"],
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
        # model_path=model_path,
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
    model_dir = "configs/model"  # model configs dir
    model_fn_base = "optuna_model_" + datetime.now().strftime('%m%d_%H%M')   # model configs file name
    log_dir = os.path.join("optuna_exp", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    model_path = os.path.join(log_dir, "best.pt")  # file path will be saved model's weight

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    global_best_f1 = 0  # 모든 trial 중 best f1, save_model에 사용

    # Create dataloader
    data_config = read_yaml('configs/data/taco.yaml')  # Hard cording
    optuna_config = read_yaml('./Special Mission/optuna_config.yaml')

    train_dl, val_dl, test_dl = create_dataloader(data_config)

    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=2)

    fig = optuna.visualization.plot_optimization_history(study)
    visualization_dir = '/opt/ml/code/visualization_result'
    fig.write_html(os.path.join(visualization_dir, f"{model_fn_base}.html"))
