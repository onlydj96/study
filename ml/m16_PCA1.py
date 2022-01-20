# PCA 

'''
PCA(Principal component analysis) : 고차원의 데이터를 저차원의 데이터로 환원시키는 기법을 말한다.
'''

from unittest import result
import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape)  # (569, 30)

pca = PCA(n_components=12)  # 칼럼이 8개의 벡터로 압축이됨
x = pca.fit_transform(x)
print(x.shape)  # (569, 12)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            train_size=0.8, random_state=66)

#2. 모델
from xgboost import XGBClassifier
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train)
result = model.score(x_test, y_test)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)


'''
1. PCA 안했을 때 
결과 :  0.9736842105263158

2. PCA 사용했을 때
결과 :  0.9473684210526315
'''