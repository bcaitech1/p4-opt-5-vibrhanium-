# Baseline code 간단 정리 <!-- omit in toc -->

- [:file_folder: File structure](#file_folder-file-structure)
- [:clipboard: Argument 종류](#clipboard-argument-종류)
  - [train](#train)
  - [inference](#inference)
- [:question: ABC 모듈은 뭘까요?](#question-abc-모듈은-뭘까요)
- [:pencil2: 코드 보충 설명](#pencil2-코드-보충-설명)
  - [src/augmentation](#srcaugmentation)
    - [src/augmentation/transforms.py](#srcaugmentationtransformspy)
    - [src/augmentation/methods.py](#srcaugmentationmethodspy)
    - [src/augmentation/policies.py](#srcaugmentationpoliciespy)
  - [src/modules](#srcmodules)
    - [src/modules/base_generator.py](#srcmodulesbase_generatorpy)
    - [그 외](#그-외)
  - [src/utils](#srcutils)
    - [src/utils/data.py](#srcutilsdatapy)
    - [src/utils/torch_utils.py](#srcutilstorch_utilspy)


## :file_folder: File structure

code/    
│　   
├─ configs/ - yaml 파일을 저장하는 곳.     
│　 ├─ data/ - dataset에 관한 yaml config    
│　 │　└ ...    
│　 └─ model/ - model에 관한 yaml config       
│　　　 └ ...   
│   
├─ data/ - data에 관한 **정보**를 저장한 곳    
│　 └─ annotation.json - 1) id, 2) label, 3) data의 bbox 위치를 저장  
│   
├─ exp/ - save file 저장경로 (best model 저장)  
│   
├─ environment.yml - requirements(dependencies)를 담고 있는 파일  
├─ mypy.ini - mypy 설정파일 [링크 참조](https://www.daleseo.com/python-mypy/)   
│  
├─ src/ - 핵심 코드 (모델 정의, 데이터 증강, 데이터 로드, 학습 등)   
│　 ├─ [augmentation/](#srcaugmentation)  
│　 │　 ├─ [methods.py](#srcaugmentationmethodspy)  
│　 │　 ├─ [policies.py](#srcaugmentationpoliciespy)  
│　 │　 └─ [transforms.py](#srcaugmentationtransformspy)   
│　 │  
│　 ├─ [modules/](#srcmodules)  
│　 │　 ├─ [base_generator.py](#srcmodulesbase_generatorpy)  
│　 │　 ├─ bottleneck<span>.</span>py    
│　 │　 ├─ activations<span>.</span>py  
│　 │　 └─ ...  
│　 │  
│　 ├─ [utils/](#srcutils)  
│　 │　 ├─ inference_utils.py - inference 결과를 암호화해주는 듯 하다.  
│　 │　 ├─ pytransform/ - 암호화 단계에서 os/플랫폼 일치. 오직 대회용 파일인듯  
│　 │　 ├─ common<span>.</span>py - `read_yaml()`, `get_label_counts()`메소드. 구현도 간단함  
│　 │　 ├─ [data.py](#srcutilsdatapy)    
│　 │　 ├─ macs<span>.</span>py - MACs를 계산, `ptflops` 라이브러리 이용    
│　 │　 └─ [torch_utils.py](#srcutilstorch_utilspy)  
│　 │  
│　 ├─ dataloader<span>.</span>py - dataloader 반환. `src/augmentation` 의 클래스 활용  
│　 ├─ loss<span>.</span>py - loss 함수들. 추후 커스터마이징 가능  
│　 ├─ model<span>.</span>py - 모델 생성. `src/modules`의 클래스 활용    
│　 └─ trainer<span>.</span>py - 학습을 위한 `TorchTrainer` 클래스 선언  
│  
├─ tests/ - debugger  
│　 └─ test_model_parser.py - 디버깅(테스팅)을 위한 클래스 구현  
│　  
├─ train<span>.</span>py - 필요한 모듈등을 종합적으로 불러와서 학습 진행  
└─ inference<span>.</span>py - 저장된 모델로 inference 진행  
  
## :clipboard: Argument 종류
### train
- `--model` => model yaml 파일 경로
- `--data` => data yaml 파일 경로

### inference
- `--dst` => `submission.csv`를 저장할 경로
- `--weight` => 저장된 모델 정보가 있는 경로(`exp/---`)
- `--model_config` => model yaml 파일 경로. best model도 yaml이 동일하니, `exp/---`의 `model.yml`을 입력해도 될 것 같네요
- `--data_config` => data yaml 파일 경로. 위와 동일
- `--img_root` => 이미지 데이터가 들어있는 경로

## :question: ABC 모듈은 뭘까요? 
현재 코드에서 클래스 선언시에 ABC 모듈이 많이 활용되고 있습니다.
요약하면, 구현되지 않은 메소드(추상 메소드)를 가진 클래스를 상속받을 경우, 해당 메소드를 구현할 수 있도록 강제화하는 모듈입니다. **즉 더욱 안전한 추상화를 돕는 모듈입니다!**    
       
혹시 추상화의 개념을 잘 모르시면 개념을 정확히 하고 넘어가시기 바랍니당  
실제로 여러 선배들께 면접에서도 많이 물어보는 개념이라고 들었어요!      
  
ABC 모듈 자체에 대한 내용은 [링크](https://bluese05.tistory.com/61) 보시면 쉽게 정리됩니다.  
 **ABC를 사용하지 않으면 특정 클래스가 실행될 때 미구현 메소드가 있음을 확인하지만(뒤늦게 error raise), ABC를 사용하면 처음 코드 실행 단계에서 패키지 내의 모든 클래스가 추상 메소드를 잘 구현했는지 미리 체크하는 것 같습니다.**

## :pencil2: 코드 보충 설명
### src/augmentation
image augmentation 관련 내용을 담고 있는 디렉토리

#### src/augmentation/transforms.py
- `torchvision.transforms` 혹은 `albumentation` 안쓰고 각종 augmentation들이 `PIL`로 직접 구현되어있습니다.
- 다만 여기서 핵심은 `transforms_info()` 메소드인데, 여기서 augmentation을 위한 딕셔너리를 반환해주므로 더 많은 augmentation을 활용하고싶다면 여기서 추가해주어야 할 것 같습니다.

#### src/augmentation/methods.py
`src/augmentation/methods.py`에서는 `transforms_info()` 메소드를 이용해 실제로 augmentation을 적용할 수 있는 클래스들이 정의되어있습니다. 
- `SequentialAugmentation`에서는 각 augmentation을 사용자가 넘겨준 확률(`policies`에 기반하여 적용합니다.
- `RandAugmentation`에서는 `n_select`개 만큼의 augmentation을 무작위로 뽑아와서 적용합니다.

#### src/augmentation/policies.py
`simple_augment_train()`, `simple_augment_test()`는 일반적인 augmentation들이고, `randaugment_train()`메소드는 앞서 `transforms.py`에서 정의된 `RandAugmentation` 클래스를 이용하여 random augmentation을 적용합니다. 다만 여기서도 `SequentialAugmentation` 클래스를 활용하는데, 넘겨주는 argument가 **Cutout 한 개 뿐이라** 이 부분은 베이스라인 기준 0.8의 확률로 강도 9의 cutout만을 적용한다고 보면 될 것 같습니다.

### src/modules
model의 module block 관련 내용을 담고 있는 디렉토리

#### src/modules/base_generator.py
model yaml 파일에서 정의한 모델을 직접 만들어주는 부분입니다.
같은 블록을 `n`개 쌓고 싶다고 선언하면 `n`개 쌓아주고, 블록들의 in channel과 out channel이 서로 일치하도록 해줍니다.

#### 그 외
`src/modules/base_generator.py`에서 블록을 쉽게 generating할 수 있도록 각 모듈을 정의하는 `src/modules`의 모든 `py` 파일들에는 `---Generator`클래스가 존재합니다. `src/modules/base_generator.py`의 `ModuleGenerator` 클래스는 사용자가 요청한 각 모듈 블록 이름을 통해 해당 블록의 `Generator` 클래스를 받아와 추후 `src/model.py`에서 모델을 정의할 때 이를 return해줍니다.
`GeneratorAbstract`는 앞서 말한 각 모듈블록의 `---Generator` 클래스의 부모 클래스입니다.

### src/utils
#### src/utils/data.py
별로 중요한 부분은 아닙니다. 일단 이런게 있다는것만 알고가면 좋을 것 같아요
- `get_rand_bbox_coord()`는 random한 bbox 좌표를 만들어 반환해줍니다. `src/augmentation/transforms.py`의 `Cutout()` 구현 시 활용합니다.
- `weights_for_balanced_classes()`는 data imbalance 문제 해결을 위해 클래스별 가중치를 주는 메소드입니다. 다만 구현이 덜되어있고, 실제로 활용하고 있지는 않아서 이걸 활용하려면 구현을 완성시킨 후 loss 주는 단계에서 활용하면 될 것 같습니다.

#### src/utils/torch_utils.py
- `split_dataset_index()`는 train/valid로 데이터셋을 나누어주는 역할을 합니다.
- `save_model()`은 학습된 모델을 저장하는 역할을 합니다.
- `model_info()`는 현재 생성된 모델을 summary해주는 역할을 합니다. parameter 수, freeze되지 않은 파라미터 수(즉, 역전파가 전달되는 파라미터 수), module list, model summary 등을 print합니다.
- `check_runtime()`은 말 그대로 실행시간을 측정합니다. default로 이미지 1개를 모델에 넣었을 때 걸리는 시간을 측정하며, 여러번(여기서는 20번) 측정하고 평균 실행시간을 반환합니다. 실제로 train/inference에서 사용되는 메소드는 아닙니다.
- `make_divisible()`은 CNN이 가진 kernel의 채널이 8의 배수인지 확인합니다. 이 메소드는 `src/modules/invertedresidualv3.py`에서 squeeze하는 단계에 호출되는데, 정확히 구현이 어떤지는 저도 몰라서 이걸 왜 쓰는지는 잘 모르겠습니다. 
- `autopad()`는 Conv layer에서 사용자가 padding 값을 주지 않을 때 호출되며 padding을 자동으로 넣어주는 용도로 활용됩니다. 
- `Activation` 클래스는 string으로 넘어온 activation 이름을 실제 사용할 수 있는 activation 클래스로 변환해줍니다. (docstring에도 나와있음)