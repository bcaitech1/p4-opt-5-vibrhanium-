# Special Mission 1, 2

## Mission 1
### 내용
이번 Search는 Optuna라는 API를 활용하여 수행됩니다. Optuna 공식 홈페이지의 Tutorial과 API 문서를 확인하여, MNIST 데이터셋 또는 CIFAR10 데이터셋, Toy model에 hyperparamter(batch_size, epochs, Learning rate 등)를 search하는 코드를 작성하고, Acc를 maximize하는 configuration을 찾아보세요. Optuna의 전체 파이프라인을 파악하고자하는 의도입니다!

### 풀이
[코드 첨부](./special_1.ipynb)  

### 눈여겨볼 점
- 발표할 때는 `nn.CrossEntropyLoss`를 분리해서 활용해야한다고 말씀드렸었는데, 현재 버전에서는 굳이 그러지 않아도 된다고 합니다! 따라서 해당 부분은 `nn.CrossEntropyLoss`를 활용하는 코드로 변경하였습니다. 다만, 아까 말씀드렸듯이 `nn.CrossEntropyLoss`는 `LogSoftmax`와 `NLLLoss`를 하나의 클래스에서 구현한 형태가 맞기 때문에 코드를 어떻게 작성하셔도 상관은 없습니다.
    - [공식 문서 참조](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)  
    > This criterion combines LogSoftmax and NLLLoss in one single class.  
  
- 이 코드를 활용하면 early stopping이 되어 `n_trials=100`을 채우지 않고 코드가 일찍 종료되는데, 그 이유는 `study.optimize` 호출 시 `timeout`이 설정되어있기 때문입니다.
    - 여기서의 `timeout`은 우리가 일반적으로 아는, 일정 시간동안 반응이 없으면 종료하는 timeout이 아니라 **말그대로 '이 시간동안 optimize를 진행했으면 종료해라'라는 의미**입니다.
    - 즉, `n_trials`이 아무리 크다고 한들, `timeout`만큼의 시간이 지나면 최적화는 종료됩니다. 현재 코드에서는 10분(600)으로 설정되어있습니다.
    - [study.optimize 공식문서 참조](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize)

- Mission 1과 같이 단일 변수에 대한 optimizing을 할 때는, `optuna` 내부에서 single 3강에서 배운 `TPESampler`를 활용한다고 합니다.
    - [create_study 공식문서 참조](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study), sampler 부분 보시면 됩니다.
  
- 현재 코드와 같이 `create_study`에서 pruner를 따로 지정하지 않을 경우 `MedianPruner`가 default pruner로 들어가게됩니다.
    - [create_study 공식문서 참조](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study), pruner 부분 보시면 됩니다.
    - `MedianPruner`는 현재 돌리고 있는 모델의 가장 좋은 결과값이 **이전 trial들에서의 결과의 중간값을 넘지 못할 경우** 더이상 학습/평가를 하지 않고 중단(prune)합니다.
    - 지금의 경우 loss 값을 기준으로 최적화를 하고 있으므로, 같은 step에서 다른 trial들의 loss 집합의 중간값이 현재 trial의 가장 낮은 loss보다 낮을 경우 중단합니다. 
    - `trial.should_prune()`에 걸리는 경우는 위와 같은 경우라고 보시면 되며, 이 경우 `state`가 `TrialState.PRUNED`로 설정되어 `pruned_trials`에 해당 trial이 들어가게 됩니다.
    - [MedianPruner 공식문서 참조](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html#optuna.pruners.MedianPruner)
    - 우리가 아는 pruning과는 조금 다른, early stopping 같은 개념이라고 보면 될 것 같습니다.
  

## Mission 2
### 내용
실제 문제에서 Search는 단일 변수가 아닌 여러 변수에 대하여 수행하는 경우가 많습니다. Optuna에서는 Multi objective study라는 기능을 통하여 이를 구현하고 있는데요. Mission1 에서 구현한 모델에 파라미터 갯수는 최소화(Minimize), 성능(Acc)는 최대화(Maximize)하는 Hyperparater configuration을 찾는 코드를 작성해보세요.

### 풀이
[코드 첨부](./special_2.ipynb)  

### 눈여겨볼 점
- 이전과 코드 내용은 완전히 동일하지만, 여기서는 `nn.CrossEntropyLoss()`대신 앞서 설명한 방식(`nn.LogSoftmax` + `F.nll_loss`)을 활용하였습니다. 역시 동작은 동일합니다.
- 또한 `train`과 `evaluation` 기능을 함수로 분리하였습니다. 역시 기능은 동일합니다.
- `flops, _ = thop.profile(model, inputs=(torch.randn(1, 28 * 28).to(device),), verbose=False)`에서는 `thop` 라이브러리의 `profile` 메소드를 활용하여 가상의 input을 넣어주고 이에 대한 연산량을 계산하여 `flops`라는 변수로 반환합니다.
    - `_`로 표시된 부분은 `parameter`를 반환받는 부분인데, 여기서는 사용하지 않습니다.
- Multi objective study에서는 이전과 모두 같지만, return하는 값이 2개 이상입니다.
- 현재 코드의 경우 `flops`와 `accuracy`를 `objective` function에서 return하고 있으므로 이 두 값이 최적화 대상이 됩니다.
- return하는 순서에 맞추어 `study = optuna.create_study(directions=["minimize", "maximize"])`와 같이 **최적화 방향**을 설정합니다.
- 이전과 달리 argument가 `direction`이 아닌 `directions`로 들어간다는 점 짚고 넘어가면 좋을 것 같습니다.