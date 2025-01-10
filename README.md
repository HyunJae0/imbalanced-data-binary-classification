## 개인 프로젝트
클래스(class)가 다소 불균형한 이진 분류 문제를 해결하기 위해 다음과 같은 식을 이용하여 최적의 임곗값(threshold)를 찾은 다음, 이 임곗값을 적용해 예측(분류)을 수행했습니다.

<img src="./img/optimal_threshold.JPG" width="50%" height="50%">

## 문제 정의
이진 분류 문제에서 일반적으로 모델이 제공한 점수를 임곗값을 적용하여 예측을 수행합니다. 

Python의 scikit-learn 라이브러리에서 모델 인스턴스의 predict( ) 메서드를 호출할 때, 이 임곗값은 보통 기본값 0.5로 설정됩니다.

0.5는 1일 확률이 0일 확률보다 크다는 것을 보장하는 자연스러운 임곗값입니다.

그러나 이것은 단순한 가정이기 때문에, 데이터에 기반하여 적절한 임곗값을 설정해야 합니다. 즉, 데이터에 따라 0.5가 적합한 임곗값이 아닐 수도 있습니다.

특히, 불균형한 비율의 클래스를 가지는 데이터의 경우 특정 클래스에 대한 데이터가 많으므로, 전체 정확도는 높게 나올 수 있으나 데이터가 적은 클래스에 대한 재현율(recall)이나 정밀도(precision) 값이 매우 낮게 나타날 수 있습니다. 이는 분류  문제에 있어 바람직한 결과가 아닙니다.

불균형 데이터의 문제를 해결하는 방법은 다양합니다. 대표적으로 다양한 업샘플링(Upsampling)과 다운샘플링(Downsampling) 기법들이 있습니다. 

다만, 다운샘플링은 데이터를 축소하기 때문에 정보 손실이, 업샘플링은 주어진 데이터 셋에 대한 성능이 향상될 수는 있으나, unseen data에 대한 예측 성능이 좋지 않을 수 있습니다. 

그러므로 샘플링 기법을 사용하는 대신, 데이터에 기반하여 클래스 불균형 문제를 완화하고자 합니다.

## Stack
```
Python
PyCaret
Keras  
TensorFlow  
```

# 코드 실행
## 1. Data preprocessing
아래 Jupyter Notebook markdown으로 기술되어 있음
https://github.com/HyunJae0/imbalanced-data-binary-classification/blob/main/preprocessing.ipynb

### 1.1 데이터 설명 및 변수 선정
사용한 데이터는 통계청 MDIS(Microdata Integrated Service)에서 제공받은 '2020년 주거실태조사' 데이터입니다. 해당 데이터는 설문 조사 응답자 46,976명에 대한 인구통계학적 정보와 주택과 관련된 정보입니다.

예측에 사용한 종속 변수는 '공공임대주택 입주의향(0, 1)'이며, 독립 변수는 관련 연구인 '공공임대주택 입주의향 영향요인에 관한 연구'에서 사용한 변수를 사용했습니다. 해당 연구는 이항 로짓 회귀분석을 이용해 어떤 요인이 공공임대주택 입주의향에 영향을 미치는지 분석을 진행하였습니다.
<img src="./img/optimal_threshold.JPG" width="50%" height="50%">


