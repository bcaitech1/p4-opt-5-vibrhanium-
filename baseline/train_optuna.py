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
        nc_args = []
        nc = trial.suggest_categorical(
            name=f"normal_cell_{i}",
            choices=["Conv", "DWConv", "Bottleneck", "InvertedResidualv2"]
            # choices=["Conv"]
            )

        if nc == "Conv":
            # Conv args: [out_channels, kernel_size, stride, padding, groups, activation]
            out_channels= trial.suggest_int(f"normal_cell_{i}/out_channel", 3, 5)
            kernel_size = trial.suggest_int(f"normal_cell_{i}/kernel_size", 3, 5, step=2)
            activation = trial.suggest_categorical(
                f"normal_cell_{i}/activation", ["ReLU", "ReLU6", "Hardswish"]
            )

            stride=1
            padding=1
            groups=1
            nc_args = [out_channels, kernel_size, stride, padding, groups, activation]

        elif nc == "DWConv":
            out_channels= trial.suggest_int(f"normal_cell_{i}/out_channel", 3, 5)
            kernel_size = trial.suggest_int(f"normal_cell_{i}/kernel_size", 3, 5, step=2)
            activation = trial.suggest_categorical(
                f"normal_cell_{i}/activation", ["ReLU", "ReLU6", "Hardswish"]
            )

            stride=1
            padding=1
            nc_args = [out_channels, kernel_size, stride, padding, activation]
        
        elif nc == "Bottleneck":
            out_channels= trial.suggest_int(f"normal_cell_{i}/out_channel", 3, 5)
            shortcut = trial.suggest_int(f"normal_cell_{i}/shortcut", 0, 1)  # 1==True, 0==False
            expansion = trial.suggest_int(f"normal_cell_{i}/expansion", 1, 3)
            activation = trial.suggest_categorical(
                f"normal_cell_{i}/activation", ["ReLU", "ReLU6", "Hardswish"]
            )

            groups = 1
            nc_args = [out_channels, shortcut, groups, expansion, activation]
        
        elif nc == "InvertedResidualv2":
            out_channels= trial.suggest_int(f"normal_cell_{i}/out_channel", 3, 5)
 
            expand_ratio = 0.5
            stride=1
            nc_args = [out_channels, expand_ratio, stride]
        
        # elif nc == "MBConv":
        #     out_channels= trial.suggest_int(f"normal_cell_{i}/out_channel", 3, 5)
        #     kernel_size = trial.suggest_int(f"normal_cell_{i}/kernel_size", 3, 5, step=2)

        #     expand_ratio = 1.0
        #     stride=1
        #     nc_args = [out_channels, expand_ratio, stride, kernel_size]

        ncx[i] = nc
        ncx_args[i] = nc_args

    nc1, nc2, nc3 = ncx
    nc1_args, nc2_args, nc3_args = ncx_args

    # Sample Reduction Cell(RC)
    n_rc = 2
    rcx = [[] for _ in range(n_rc)]
    rcx_args = [[] for _ in range(n_rc)]
    for i in range(n_rc):
        rc = trial.suggest_categorical(
            f"reduction_cell_{i}",
            ["InvertedResidualv2", "InvertedResidualv3", "MaxPool", "AvgPool"]
            # ["InvertedResidualv2"]
            )
        if rc == "InvertedResidualv2":
            out_channels = trial.suggest_int(f"reduction_cell_{i}/out_channels", 3, 5)

            expand_ratio = 0.5
            stride = 2
            rc_args = [out_channels, expand_ratio, stride]

        elif rc == "InvertedResidualv3":
            kernel_size = trial.suggest_int(f"normal_cell_{i}/kernel_size", 3, 5, step=2)
            expansion = trial.suggest_int(f"normal_cell_{i}/expansion", 1, 3)
            out_channels = trial.suggest_int(f"reduction_cell_{i}/out_channels", 3, 5)
            use_se = trial.suggest_int(f"normal_cell_{i}/use_se", 0, 1)  # 1==True, 0==False
            use_hs =  trial.suggest_int(f"normal_cell_{i}/use_hs", 0, 1)  # 1==True, 0==False

            stride = 2
            rc_args = [kernel_size, expansion, out_channels, use_se, use_hs, stride]

        elif rc == "MaxPool":
            kernel_size = trial.suggest_int(f"normal_cell_{i}/kernel_size", 3, 5, step=2)

            stride = 2
            padding = 0
            rc_args = [kernel_size, stride, padding]

        elif rc == "AvgPool":
            kernel_size = trial.suggest_int(f"normal_cell_{i}/kernel_size", 3, 5, step=2)

            stride = 2
            padding = 0
            rc_args = [kernel_size, stride, padding]


        rcx[i] = rc
        rcx_args[i] = rc_args

    rc1, rc2 = rcx
    rc1_args, rc2_args = rcx_args

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
