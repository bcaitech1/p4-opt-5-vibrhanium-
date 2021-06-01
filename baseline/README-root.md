# 비브랴늄 <!-- omit in toc -->
가볍고 단단한(robust한) 모델을 목표로 합니다.
 
- [Overview](#overview)
- [File structure](#file-structure)
- [Config](#config)
  - [Data](#data)
  - [Model](#model)
- [Usage](#usage)
  - [Docker](#docker)
  - [Train](#train)
  - [Inference](#inference)

## Overview
최근들어 분야를 막론하고 인공지능 기술은 사람을 뛰어넘은 엄청난 성능을 보여주고 있고, 때문에 여러 산업에서 인공지능을 이용해 그동안 해결하지 못 했던 문제들을 풀려는 노력을 하고 있습니다.

대표적인 예로는 수퍼빈의 수퍼큐브가 있습니다. 수퍼큐브는 수퍼빈에서 만든 인공지능 분리수거 기계로 사람이 기계에 캔과 페트병을 넣으면 내부에서 인공지능을 통해 재활용이 가능한 쓰레기인지를 판단해 보관해주는 방식입니다. 간단한 인공지능을 이용해 그동안 힘들었던 분리수거 문제를 해결한 것입니다. 그렇다면 수퍼큐브를 만들기 위해 필요한 인공지능은 무엇일까요? 당연히 들어온 쓰레기를 분류하는 작업일 것입니다. 하지만 분류만 잘 한다고 해서 사용할 수 있는 것은 아닙니다. 로봇 내부 시스템에 탑재되어 즉각적으로 쓰레기를 분류할 수 있어야만 실제로 사용이 될 수 있습니다.

이번 프로젝트를 통해서는 분리수거 로봇에 가장 기초 기술인 쓰레기 분류기를 만들면서 실제로 로봇에 탑재될 만큼 작고 계산량이 적은 모델을 만들어볼 예정입니다.

## File structure
```
input/
│
├── Dockerfile
├── LICENSE.md
├── Makefile
├── README.md
├── configs
│   ├── data
│   │   ├── cifar10.yaml
│   │   └── taco.yaml
│   └── model
│   │   ├── example.yaml
│   │   └── mobilenetv3.yaml
│   └── optuna_config
│   │   ├── base_config.yaml
│   │   └── module_config.yaml
│   │   └── optimizer_config.yaml
├── environment.yml
├── inference.py
├── mypy.ini
├── src
│   ├── init.py
│   ├── augmentation
│   │   ├── methods.py
│   │   ├── policies.py
│   │   └── transforms.py
│   ├── dataloader.py
│   ├── loss.py
│   ├── model.py
│   ├── modules
│   │   ├── init.py
│   │   ├── activations.py
│   │   ├── base_generator.py
│   │   ├── bottleneck.py
│   │   ├── conv.py
│   │   ├── dwconv.py
│   │   ├── flatten.py
│   │   ├── invertedresidualv2.py
│   │   ├── invertedresidualv3.py
│   │   ├── linear.py
│   │   ├── mbconv.py
│   │   └── poolings.py
│   ├── trainer.py
│   └── utils
│   ├── common.py
│   ├── data.py
│   ├── inference_utils.py
│   ├── macs.py
│   ├── pytransform
│   │   └── init.py
│   └── torch_utils.py
├── tests
│   └── testmodelparser.py
└── train.py
```

## Config
### Data
데이터셋과 학습에 관련된 파라미터를 저장하는 설정 파일입니다. 
- Example
```
DATA_PATH: "/opt/ml/input/data/" # 데이터셋이 위치한 경로 입니다.
DATASET: "TACO" # 데이터셋의 이름 입니다.
IMG_SIZE: 224  # 이미지를 224x224 로 리사이징 합니다. 필요에 따라 조정이 가능합니다.
AUG_TRAIN: "randaugment_train"  # 학습셋에 대한 Augmentation 정책입니다.
AUG_TEST: "simple_augment_test"  # Validation셋에 대한 Augmentation 정책 입니다.
AUG_TRAIN_PARAMS:  # 학습셋에서 사용하는 Augmentation 정책의 파라미터 입니다.
    n_select: 2  # 동시에 몇 개의 Augmentation 이 적용될지를 선택합니다.
    level: null  # Augmentation의 강도를 설정 합니다. (null 일 경우 기본 값을 사용 합니다.)
AUG_TEST_PARAMS: null  # Validation셋에서 사용하는 Augmentation 정책의 파라미터 입니다. (null 일 경우 기본 값을 사용 합니다.)
BATCH_SIZE: 256  # Batch size 입니다.
EPOCHS: 200  # 학습 총 epochs 를 설정 합니다.
VAL_RATIO: 0.1  # 학습셋 안에서 Validation의 비율을 조정 합니다.​
```

### Model
모델을 구성하는 설정 파일이 위치합니다. 다양한 모델들을 yaml 파일로 구성할 수 있습니다.

**구성요소**
- input_channel: 입력으로 들어오는 채널 수 입니다. RGB 이미지의 경우 3이 됩니다.
- depth_multiple: 반복 횟수의 곱셈 값입니다.
- width_multiple: 채널 갯수의 곱셈 값입니다.
- backbone: 모듈을 포함하는 리스트 입니다. 각 모듈에 대한 설명은 아래에서 설명하겠습니다.
    - module: 모듈은 총 3개의 파라미터로 설정이 됩니다. [반복 횟수, 모듈 이름, 모듈 인자값]
        - 반복 횟수: 해당 모듈을 몇 번 반복할지 결정 합니다. 최종 반복 횟수는 max(round(반복 횟수 * depth_multiple), 1) 입니다. * 반복 횟수가 1인 경우는 1로 고정 됩니다.
        - 모듈 이름: 모듈 이름입니다. 지원하는 종류는 (Bottleneck, Conv, DWConv, Linear, GlobalAvgPool, InvertedResidualv2, InvertedResidualv3, FixedConv, Linear, DWConv, Flatten, MaxPool, AvgPool, GlobalAvgPool) 입니다. 필요할 경우 학습자가 직접 모듈을 만드는 것도 가능합니다.
        - 모듈 인자값: 각 모듈에서 사용하는 모듈의 인자값을 설정 합니다. 각 모듈별 인자값을 아래에 자세히 설명하겠습니다.

- Example
```
input_channel: 3  # 입력으로 들어오는 채널 수 입니다. RGB 이미지의 경우 3이 됩니다.

depth_multiple: 1.0  # 반복 횟수의 곱셈 값입니다. 
width_multiple: 1.0  # 채널 갯수의 곱셈 값입니다.

backbone:
		# 파이토치 기본 튜토리얼을 그대로 재구성한 모델입니다.
    # PyTorch Tutorial (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
    # [repeat, module, args]
    [
        [1, Conv, [6, 5, 1, 0]],  # 6채널, 5x5 커널 사이즈, stride 1, 패딩 0
        [1, MaxPool, [2]],  # 2x2 Max Pooling
        [1, Conv, [16, 5, 1, 0]],
        [1, MaxPool, [2]],
        [1, GlobalAvgPool, []],
        [1, Flatten, []],
        [1, Linear, [120, ReLU]],
        [1, Linear, [84, ReLU]],
        [1, Linear, [10]]
    ]​
```

## Usage
### Docker
```
docker run -it --gpus all --ipc=host -v $PWD:/opt/ml/code -v ${dataset}:/opt/ml/data placidus36/pstage4_lightweight:v0.1 /bin/bash
```

### Train
`train.py` 실행
```
python train.py --model ${모델 파일 경로} --data ${데이터셋 파일 경로}
# Ex) python train.py --model configs/model/mobilenetv3.yaml --data configs/data/taco.yaml
```

`train_optuna.py` 실행
```
python train_optuna.py
# Ex) python train.py
```

### Inference
`inference.py` 실행
```
python inference.py --model_config ${모델 yaml 경로} --weight ${모델 weight 경로} --img_root ${test 데이터 경로} --data_config ${데이터 yaml 경로}
# EX) python inference.py --model_config configs/model/mobilenetv3.yaml --weight exp/2021-05-13_16-41-57/best.pt --img_root /opt/ml/input/test/ --data_config configs/data/taco.yaml
```
