import argparse
from typing import Any, Dict, Union

import yaml
import numpy as np
from torchvision.datasets import ImageFolder, VisionDataset


def read_yaml(cfg: Union[str, Dict[str, Any]]):
    if not isinstance(cfg, dict):
        with open(cfg) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = cfg
    return config


def get_label_counts(dataset_path: str):
    """Counts for each label."""
    if not dataset_path:
        return None
    td = ImageFolder(root=dataset_path)
    # get label distribution
    label_counts = [0] * len(td.classes)
    for p, l in td.samples:
        label_counts[l] += 1
    return label_counts


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_LBscore(f1, macs): # 낮을수록 좋음
    # 지금 식으로는 optuna를 최적화할 수 없다.
    # if f1 < 0.5:  
    #     score = 1.0
    # elif f1 < 0.8342: 
    #     score = (1 - f1 / 0.8342)
    # else:
    #     score = 0.5 * (1 - f1 / 0.8342)
    if f1 < 0.8342: 
        score = (1 - f1 / 0.8342)
    else:
        score = 0.5 * (1 - f1 / 0.8342)
    score += macs / 13863553
    return score
