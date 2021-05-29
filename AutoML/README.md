## AutoML(Automated Machine Learning)
AutoML을 통해 어느정도의 성능을 내면서, 연산량은 작은 모델을 탐색해본다.
<br/>:arrow_forward: [[Pstage] 모델최적화 - 3강 AutoML 이론](https://www.edwith.org/bcaitech1/lecture/782185?isDesc=false)
<br/>:arrow_forward: [[Pstage] 모델최적화 - 4강 AutoML 실습](https://www.edwith.org/bcaitech1/lecture/782186?isDesc=false)
<br/>:arrow_forward: [[Pstage] 모델최적화 - 4강 Data Augmentation 및 AutoML 실습](https://www.edwith.org/bcaitech1/lecture/782190?isDesc=false)

📑 [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)

### Special mission1 - Model Search 코드 작성하기
[`Optuna`](https://optuna.org/)
|Misson|내용|
|:---:|---|
|[**1**](https://github.com/bcaitech1/p4-opt-5-vibrhanium-/tree/jaegyeong/readme/AutoML/special_mission_1_2)|Fashion MNIST 데이터셋을 사용하여 Toy model에 hyperparamter(batch_size, epochs, Learning rate 등)를 search하는 코드 작성하여 Acc를 maximize하는 configuration을 탐색|
|[**2**](https://github.com/bcaitech1/p4-opt-5-vibrhanium-/tree/jaegyeong/readme/AutoML/special_mission_1_2)|모델의 파라미터 갯수는 최소화(Minimize), 성능(Acc)는 최대화(Maximize)하는 Hyperparater configuration을 찾는 코드를 작성|
|**3**|Optuna API를 사용하여 yaml 파일을 생성 및 search하는 코드 작성<br/>&nbsp;&nbsp;- Layer는 몇개를 둘 것인지<br/>&nbsp;&nbsp;- Layer module은 어떤 것들을 적용할 것인지<br/>&nbsp;&nbsp;- 각 Layer 위치에 따라 search space는 어떻게 구성할건지|
|**4**|AutoML의 학습 시간을 줄이는 방법에는 어떤 것들이 있을지 고민|

### Special mission2 - DB & 실험 분석 API 연동하기
[`Optuna`](https://optuna.org/) [`Postgresql`](https://www.postgresql.org/) [`wandb`](https://wandb.ai/site) [`MLflow`](https://mlflow.org/)
|Misson|내용|
|:---:|---|
|**1**|Optuna에 DB연동<br/>&nbsp;&nbsp;a) PStage에서 지급된 서버에 postgresql 직접설치/도커 이미지를 셋업하여 연동하기<br/>&nbsp;&nbsp;b) 외부 cloud 또는 개인 pc에 postgresql을 직접설치/도커 이미지로 셋업하여 연동하기|
|**2**|Baseline 코드에 실험 분석 API 연동하여 AutoML의 search space가 적절하게 설정되었는지 등의 요소들을 잘 파악할 수 있도록 다양한 configuration들을 로깅|
