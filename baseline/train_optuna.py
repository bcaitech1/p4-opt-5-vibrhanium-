import os
import yaml
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

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
    # 초기 설정 setting
    num_cells = base_config['num_cells']
    normal_cells = module_config['normal_cells']
    reduction_cells = module_config['reduction_cells']
    model_config = {'input_channel': 3,
  'depth_multiple': 1.0,
  'width_multiple': 1.0,
  'backbone': [],}
    
    # Sample Normal Cell(NC)
    n_nc = num_cells['value'] 

    low, high = num_cells['low'], num_cells['high']
    # deeper layer, more features
    nx = [trial.suggest_int(name=f"n{i}_repeat", low=low*(i), high=high*(i)) for i in range(1,n_nc+1)] 
    
    ncx = [[] for _ in range(n_nc)]
    ncx_args = [[] for _ in range(n_nc)]
    for i in range(n_nc):
        nc = suggest_from_config(trial, base_config, 'normal_cells', idx=i)
        nc_config = normal_cells[nc]

        nc_args = []
        for arg, value in nc_config.items(): #out_channel, {name:oc, type:int ... } | kernel_size, {asd}
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
        rc_config = reduction_cells[rc]

        rc_args = []
        for arg, value in rc_config.items():
            if isinstance(value, dict):
                rc_args.append(suggest_from_config(trial, rc_config, arg, idx=i))
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
    model_config["backbone"].append([1, "Linear", [num_class]])

    return model_config


def train_model(trial,
    model_instance: nn.Module
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
    epochs = suggest_from_config(trial, base_config, 'epochs')
    batch_size = suggest_from_config(trial, base_config, 'batch_size')
    
    # Sample optimizer
    optimizer_name = suggest_from_config(trial, base_config, 'optimizer') ## AdamW
    
    # 좀 더 간단하게! 할 수 있을거야 #############
    opt_params = optimizer_config[optimizer_name].keys() #lr, beta1, beta2
    temp_dict = {}
    for p in opt_params: # lr, betas
        if p == 'betas':
            beta1 = suggest_from_config(
                trial, optimizer_config[optimizer_name], p
            )  
            beta2 = optimizer_config[optimizer_name][p]['beta2']
            temp_dict['betas'] = (beta1, beta2)
            continue

        temp_dict[p] = suggest_from_config(
            trial, optimizer_config[optimizer_name], p
        )

    optimizer = getattr(optim, optimizer_name)(
        model_instance.parameters(), **temp_dict
    )  # dictionary unpacking
    
    # scheduler
    scheduler_name = suggest_from_config(trial, base_config, 'scheduler') ## CosineAnnealingLR
    
    sch_params = scheduler_config[scheduler_name].keys() # T_max eta_min last_epoch
    temp_dict = {}
    for s in sch_params:
        temp_dict[s] = suggest_from_config(trial, scheduler_config[scheduler_name], s) # lr, beta1, beta2
    
    scheduler = getattr(lr_scheduler, scheduler_name)(optimizer=optimizer, **temp_dict) # dictionary unpacking
    
    # criterion
    criterion_name = suggest_from_config(trial, base_config, 'criterion') 
    criterion = getattr(nn, criterion_name)()

    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )
    
    data_config['IMG_SIZE'] = suggest_from_config(trial, base_config, 'img_size')
    data_config['BATCH_SIZE'] = batch_size
    train_dl, val_dl, test_dl = create_dataloader(data_config)
    
    # Create trainer
    trainer = TorchTrainer(
        model=model_instance.model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        # model_path=model_path,
        verbose=1,
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

    model_instance = Model(model_config, verbose=True)
    model_instance.model.to(device)

    # Calc macs
    macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
    print(f"macs: {macs}")

    best_f1 = train_model(trial,model_instance)

    return best_f1, macs


if __name__ == '__main__':
    # Setting directory and file name
    cur_time = datetime.now().strftime('%m%d_%H%M')

    # for save best trials model config
    save_config_dir = "configs/optuna_model"
    save_config_fn_base = cur_time
    os.makedirs(save_config_dir, exist_ok=True)

    # for save model weight
    save_model_dir = "optuna_exp"
    save_model_path = os.path.join(save_model_dir, f"{cur_time}_best.pt")
    os.makedirs(save_model_dir, exist_ok=True)
    
    # for save visualization html
    visualization_dir = '/opt/ml/code/visualization_result'
    os.makedirs(visualization_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_class = 9

    # Create config file
    data_config = read_yaml('configs/data/taco_sample.yaml')
    base_config = read_yaml('configs/optuna_config/base_config.yaml')
    module_config = read_yaml('configs/optuna_config/module_config.yaml') 
    optimizer_config = read_yaml('configs/optuna_config/optimizer_config.yaml') 
    scheduler_config = read_yaml('configs/optuna_config/scheduler_config.yaml') 

    # Optuna study
    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=3)

    # Save best trials model architecture and hyper-parameter
    for i, best_trial in enumerate(study.best_trials):
        config_fn = f"{save_config_fn_base}_best_trials{i}.yaml"
        with open(os.path.join(save_config_dir, config_fn), "w") as f:
            yaml.dump(best_trial.params, f, default_flow_style=False)

    # Visualization
    fig = optuna.visualization.plot_pareto_front(study)
    fig.write_html(os.path.join(visualization_dir, f"{save_config_fn_base}_pareto_front.html"))
