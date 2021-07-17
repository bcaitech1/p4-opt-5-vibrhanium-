# [P Stage4] 모델최적화


## 📝 대회 소개
모델 경량화는 deploy 시점에 고려해야할 중요한 기법 중 하나입니다. 본 대회는 재활용 쓰레기 데이터셋에 대해서 이미지 분류를 수행하는 모델을 설계합니다. 이때, 모델의 분류 성능 이외에도 모델의 연산 횟수도 고려하여 데이터셋에 적합한 capacity의 경량 모델 여부를 추가로 평가합니다.
- 대회 제목 : “초”경량 이미지 분류기
- 문제 유형 : muti-classfication and lightweight
- 평가척도 : F1-Score + MACs: 모델의 (덧셈 + 곱셈)의 계산 수
모델의 f1 score가 0.5이상 후에 f1 score의 비중을 줄여 MACs를 통해 좋은 점수를 받을 수 있도록 score를 설계



$score = score_{MACs} + socre_{F1}$

$$score_{MACs} = \frac {제출모델 MACs} {기준모델 MACs}$$

$$ score_{F1}=\begin{cases}1\; \quad \quad\quad\quad\quad\quad\quad\quad\quad\quad,if   \ 제출모델 F1score < 0.5 \;\\0.5*(1- \frac {제출모델 F1score}{기준모델 F1score})\;,if \ 제출모델 F1score \ ≥ 0.5\end{cases}$$

	
## 🎁 데이터 소개 (데이터 비공개)
COCO format의 재활용 쓰레기 데이터인 TACO 데이터셋의 Bounding box를 crop 한 데이터
- 총 이미지 : 32,599
- 카테고리 수 : 9개( Battery, Clothing , Glass, Metal, Paper, Paperpack, Plastic, Plasticbag, Styrofoam)
- 이미지당 크기: 고정 되어 있지 않음
- 한 카테고리당 사진의 개수: 그림 첨부 밑에 
<p align="center"><img src="https://user-images.githubusercontent.com/31814363/126025761-e6d1aec1-e123-4a5d-bb88-edd296468365.png" width="450" height="450"></p>
 

### 모델 경량화를 하는 두가지 접근 방법
#### :triangular_flag_on_post: (주어진) 모델을 경량화
기존 가지고 있는 모델을 경량화 하는 방법으로 대부분의 방법이 후처리(마스킹된 layer들을 없애는 과정 등)를 필요로 한다.
- [Pruning](https://github.com/bcaitech1/p4-opt-5-vibrhanium-/tree/master/Pruning)
- [Tensor decomposition](https://github.com/bcaitech1/p4-opt-5-vibrhanium-/tree/master/Tensor_decomposition)

#### :triangular_flag_on_post: (새로운) 경량 모델 탐색
Search를 통하여 경량 모델을 찾는 기법으로 조금 더 일반적인 적용이 가능하다.
- [NAS(Neural Architecture Search)](https://github.com/bcaitech1/p4-opt-5-vibrhanium-/tree/master/NAS)
- [AutoML(Automated Machine Learning)](https://github.com/bcaitech1/p4-opt-5-vibrhanium-/tree/master/AutoML)

