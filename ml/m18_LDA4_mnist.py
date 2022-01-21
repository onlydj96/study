
# LDA

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import warnings
warnings.filterwarnings('ignore')


#1. 데이터 정제
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, -1)  # (60000, 784)
x_test = x_test.reshape(10000, -1)  # (10000, 784)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print("LDA 전 : ", x_train.shape)

# LDA
lda = LinearDiscriminantAnalysis(n_components=-1)

lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)


#2. 모델
from xgboost import XGBClassifier
model = XGBClassifier()

#3. 훈련

model.fit(x_train, y_train, eval_metric='error')
result = model.score(x_test, y_test)


print("LDA 후 : ", x_train.shape)


#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)


'''
1. mnist
LDA 전 :  (60000, 784)
LDA 후 :  (60000, 4)
결과 :  0.8282
'''