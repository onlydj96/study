
# LDA

import numpy as np
from sklearn.datasets import fetch_covtype, load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import warnings
from sympy import Min
warnings.filterwarnings('ignore')


#1. 데이터 정제
# datasets = load_iris()
# datasets = load_breast_cancer()
# datasets = load_wine()
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            train_size=0.8, random_state=66, stratify=y)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print("LDA 전 : ", x_train.shape)

# LDA
lda = LinearDiscriminantAnalysis()

lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)


#2. 모델
from xgboost import XGBClassifier
model = XGBClassifier()

#3. 훈련

model.fit(x_train, y_train, eval_metric='merror')
result = model.score(x_test, y_test)


print("LDA 후 : ", x_train.shape)


#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)


'''
1. iris
LDA 전 :  (120, 4)
LDA 후 :  (120, 2)
결과 :  1.0

2.cancer
LDA 전 :  (455, 30)
LDA 후 :  (455, 1)
결과 :  0.9473684210526315

3. wine
LDA 전 :  (142, 13)
LDA 후 :  (142, 2)
결과 :  1.0

4. fetch_covtype
LDA 전 :  (464809, 54)
LDA 후 :  (464809, 6)
결과 :  0.7878109859470065
'''