## AutoML(Automated Machine Learning)
AutoML을 통해 어느정도의 성능을 내면서, 연산량은 작은 모델을 탐색해본다.

## Special mission - `Optuna` 사용
1. Fashion MNIST 데이터셋을 사용하여 Toy model에 hyperparamter(batch_size, epochs, Learning rate 등)를 search하는 코드를 작성하고, Acc를 maximize하는 configuration을 찾아본다.
2. 모델의 파라미터 갯수는 최소화(Minimize), 성능(Acc)는 최대화(Maximize)하는 Hyperparater configuration을 찾는 코드를 작성한다.
3. Optuna API를 사용하여 yaml 파일을 생성 및 search하는 코드를 작성해봅시다.
    - Layer는 몇개를 둘 것인지
    - Layer module은 어떤 것들을 적용할 것인지
    - 각 Layer 위치에 따라 search space는 어떻게 구성할건지
4. AutoML은 기본적으로 여러회의 학습 iteration이 필요하기 때문에, 자원과 시간이 매우 많이 투입되어야한다. 학습 시간을 줄이는 방법에는 어떤 것들이 있을지 고민해본다.
