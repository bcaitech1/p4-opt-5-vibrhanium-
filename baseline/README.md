# P stage 4 - Model optimization <!-- omit in toc -->

- [File structure](#file-structure)
- [바뀐 점](#바뀐-점)
  - [optuna_config](#optuna_config)
    - [1. base_config.yaml](#1-base_configyaml)
    - [2. module_config.yaml](#2-module_configyaml)
    - [3. optimizer_config.yaml](#3-optimizer_configyaml)
    - [4. Scheduler_config.yaml](#4-scheduler_configyaml)
  - [src/trainer.py](#srctrainerpy)
  - [optuna_search.py](#optuna_searchpy)
    - [작동 방식](#작동-방식)
    - [결과](#결과)
  - [optuna_train.py](#optuna_trainpy)
    - [작동방식](#작동방식)
    - [결과](#결과-1)
- [사용법](#사용법)
- [Reference](#reference)

**yaml 파일만으로 Custom search space 구성하여 실험을 편리하게 할 수 있도록 함**

- 모듈 block(micro)는 탐색 ⭕
- 모듈 block들의 조합 및 구성(macro) 탐색 ❌, 왼쪽 아래 구조(CIFAR10 Architecture)를 차용

    ![image](https://user-images.githubusercontent.com/71882533/120309048-babff480-c30f-11eb-98ec-847879388967.png)
    <br/>🖇️ [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)

- 주의 사항: 모듈 block의 조합 및 구성 (macro)를 탐색하기 위해서는 추가적인 수정이 필요합니다.

<기존의 방식>

`train.py` 파일 내에서 하이퍼 파라미터를 설정할 때 일일이 `trial.suggest_` 함수를 사용해서 suggestion을 설정해주어야 했다.

<수정된 방식>

`config/optuna_config`의 yaml 파일의 내용을 변경함으로써 간편하게 하이퍼 파라미터의 suggestion을 설정해줄 수 있게 되었다.

또한 가장 좋은 성능을 낸 모델 (`best.pt`), 해당 모델의 구조와 하이퍼 파라미터 (`{current_time}.yaml`), `optuna.Study`의 시각화 결과 (`_pareto_font.html`)를 저장한다.

## File structure

```
input
│
...
├── configs
│   ├── data
│   │   ├── cifar10.yaml
│   │   ├── taco.yaml
│   │   └── taco_sample.yaml
│   └── model
│   │   ├── example.yaml
│   │   └── mobilenetv3.yaml
│   └── optuna_config
│   │   ├── base_config.yaml
│   │   └── module_config.yaml
│   │   └── optimizer_config.yaml
...
├── src
│   ├── init.py
...
│   ├── trainer.py
...
│   └── torch_utils.py
├── tests
│   └── testmodelparser.py
├── train.py
├── train_optuna.py
└── optuna_search.py

```

## 바뀐 점

### optuna_config

`Anchor`를 사용해 자주 사용하는 파라미터는 쉽게 수정할 수 있도록 함

```yaml
# optimizer_config.yaml

# Anchor
lr_low      : &lr_low     0.001
lr_high     : &lr_high    0.001
beta1_low   : &beta1_low  0.9
beta1_high  : &beta1_high 0.9
weight_decay_low  : &weight_decay_low   0.01
weight_decay_high : &weight_decay_high  0.01

# optimizer config
Adam:
  lr     :
    name : lr
    type : float
    low  : *lr_low
    high : *lr_high
    log  : True

  betas  :
    name : betas
    type : float
    low  : *beta1_low
    high : *beta1_high
    beta2 : 0.9999
...
```

#### 1. base_config.yaml

모델 학습의 기본적인 파라미터들을 설정하는 yaml 파일

설정 가능한 파라미터:

- num_cells, normal_cells, reduction_cells, batch_size, epochs, optimizer, scheduler, criterion, img_size, num_layers, num_channels, num_units, dropout_rate, max_learning_rate, drop_path_rate, kernel_size

#### 2. module_config.yaml

모듈에 대한 파라미터들을 설정하는 yaml 파일

설정 가능한 파라미터:

- normal_cells:
    - Conv, DWConv, Bottleneck, InvertedResidualv2
- reduction_cells:
    - InvertedResidualv2, InvertedResidualv3, MaxPool, AvgPool

#### 3. optimizer_config.yaml

optimizer에 대한 파라미터들을 설정하는 yaml 파일

설정 가능한 파라미터:

- Adam (lr, betas), AdamW (lr, betas, weight_decay), SGD (lr)

#### 4. Scheduler_config.yaml

scheduler에 대한 파라미터들을 설정하는 yaml 파일

설정 가능한 파라미터:

- StepLR (step_size, gamma, last_epoch), 
CosineAnnealingLR (T_max, eta_min, last_epoch)

### src/trainer.py

`TorchTrainer` object 생성시 `model_path`를 인자로 전해주면 model의 weight를 저장하고 전해주지 않으면 model의 weight를 저장하지 않는다.

```python
# optuna_search.py

# Create trainer
trainer = TorchTrainer(
    model=model_instance.model,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    scaler=scaler,
    device=device,
    model_path=model_path, # KEY POINT
    verbose=1,

)
```

### optuna_search.py

#### 작동 방식

- `suggest_from_config(trial, config_dict, key, name)` 함수를 사용하여 입력된 `config.yaml` 파일에서 해당 key 인자를 가져와서 suggestion으로 바꿔줌.

#### 결과
1. 자신의 server에서 실행한 모든 trials에 해당되는 모델 architecture
    - 저장 위치: `input/config/optuna_model{mmdd_HHMM}/{i}`
    - 파일 이름: `model.yaml`
3. 자신의 server에서 실행한 모든 trials에 해당되는 hyperparameter 
    - 저장 위치: `input/config/optuna_model/{mmdd_HHMM}/{i}`
    - 파일 이름: `hyperparameter.yaml`
4. visualization된 파일
    - 저장 위치: `output/visualization_result`
    - 파일 이름: `{mmdd_HHMM}_pareto_front.html`
5. [옵션] best epoch의 model weight
    - 저장 위치: `input/config/optuna_model/{mmdd_HHMM}/{i}`
    - 파일 이름: `best.pt`
    - 주의 사항: **각 trial**에 대해서 **f1값을 기준**으로 가장 좋은 모델을 선정합니다.


### optuna_train.py

#### 작동방식

- `optuna_search.py`의 실행결과로 생성된 `model.yaml` 파일과 `hyperparameter.yaml` 파일을 입력으로 받아 학습시작
- model weight가 저장된 pt 파일을 `--weight` 인자로 줄 경우 해당 weight 부터 학습

#### 결과
1. best epoch의 model weight
    - 저장 위치: 사용된 model_config가 저장된 directory
    - 파일 이름: `train_best.pt`
    - 주의 사항: **각 trial**에 대해서 **f1값을 기준**으로 가장 좋은 모델을 선정합니다.

## 사용법

- `optuna_search.py` 실행
    - `input/config/optuna_config` 폴더 아래 yaml 파일들을 원하는 파라미터로 수정한 후 `optuna_search.py` 실행
    - `study_name` argument를 주지 않을 경우 데이터베이스 없이 로컬 메모리 데이터만으로 학습을 진행합니다.
    ```
    python optuna_search.py --n_trials ${탐색시도 횟수} \
                            --study_name ${optuna study의 별칭(이름)} \
                            --save_all ${모든 trials 저장여부} \ 
                            --save_model ${model weight 저장여부} \ 
                            --base ${base config 파일 경로} \ 
                            --module ${moduel config 파일 경로} \
                            --optimizer ${optimizer config 파일 경로} \
                            --scheduler ${scheduler config 파일 경로} \
                            --data ${data config 파일 경로}
    ```
    - 예시
    ```
    # 예시
    python optuna_search.py --n_trials 10 \
                            --study_name my_study \
                            --save_all True \ 
                            --save_model False \
                            --base configs/optuna_config/base_config.yaml
    ```
- `train.py` 실행
    ```
    python train.py --model ${모델 파일 경로} \
                    --data ${데이터셋 파일 경로}
    ```

    ```
    # 예시
    python train.py --model configs/model/mobilenetv3.yaml \
                    --data configs/data/taco.yaml
    ```
- `optuna_train.py` 실행
    - **`model` argument는 default 값이 없으므로 반드시 직접 설정해주어야 합니다.**
    - **`hyperparam` argument는 default 값이 없으므로 반드시 직접 설정해주어야 합니다.**
    - 
    ```
    python optuna_train.py --data ${데이터셋 파일 경로} \
                           --model ${모델 파일 경로} \
                           --hyperparam ${hyperparameter 파일 경로} \
                           --weight ${모델 weight 경로}
    ```

    ```
    # 예시
    python optuna_train.py --data configs/data/taco.yaml \
                           --model /opt/ml/input/config/optuna_model/0604_1244/0/0604_1244_0_model.yaml \
                           --hyperparam /opt/ml/input/config/optuna_model/0604_1244/0/0604_1244_0_hyperparameter.yaml \
                           --weight /opt/ml/output/optuna_exp/0604_1244_best.pt
    ```
- `inference.py` 실행
    - **`model_config` argument는 default 값이 없으므로 반드시 직접 설정해주어야 합니다.**
    - **`weight` argument는 default 값이 없으므로 반드시 직접 설정해주어야 합니다.**
    - **`img_root` argument는 default 값이 없으므로 반드시 직접 설정해주어야 합니다.**
    - **`data_config` argument는 default 값이 없으므로 반드시 직접 설정해주어야 합니다.**
    ```
    python inference.py --dst ${제출 파일 저장할 디렉토리} \
                        --weight ${모델 weight 경로} \
                        --model_config ${모델 yaml 경로} \ 
                        --data_config ${데이터 yaml 경로} \
                        --hyperparam ${hyperparam yaml 경로} \
                        --img_root ${test 데이터 경로} \ 
                        --decompose ${텐서 분해 여부}
    ```

    ```
    # 예시
    python inference.py --dst /opt/ml/output \
                        --weight /opt/ml/input/optuna_exp/0604_2353/70/best.pt \
                        --model_config /opt/ml/input/config/optuna_model/0604_2353/70/0604_2353_70_model.yaml \
                        --data_config configs/data/taco.yaml \
                        --hyperparam /opt/ml/input/config/optuna_model/0604_2353/70/0604_2353_70_hyperparameter.yaml \
                        --img_root /opt/ml/input/data/test/ \
                        --decompose False
    ```

## Reference

- `suggest_from_config` : [https://github.com/mbarbetti/optunapi](https://github.com/mbarbetti/optunapi)
- `YAML file anchor 사용법` : [YAML - yjol_velog](https://velog.io/@yjok/YAML#:~:text=%EC%95%B5%EC%BB%A4%EB%8A%94%20%26%20%EB%A1%9C%20%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94,%EC%9D%84%20%EC%B0%B8%EC%A1%B0%ED%95%A0%20%EC%88%98%20%EC%9E%88%EB%8B%A4)
- `Optuna tutorial` : [https://optuna.readthedocs.io/en/stable/tutorial/index.html](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
