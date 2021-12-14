
# 이진분류 sigmoid 함수 사용법

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.engine.sequential import Sequential

from sklearn.model_selection import train_test_split

'''
이진분류 모델 : loss값을 0과 1사이로 좁혀서 두 개의 카테고리를 중 하나를 예측하는 모델
'''

#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1004)

# print(datasets.DESCR)
# print(datasets.feature_names)   
# print(x.shape, y.shape) # (569. 30), (569,)

print(np.unique(y)) # y에 반환되는 값을 정리


#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=30))  # activation=linear은 default값이다
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1, activation='sigmoid')) # activation='sigmoid'를 통해 loss값을 0과1 사이로 한정시킨다.

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

'''
분류모델에의 컴파일
binary_crossentropy : 낮은 확률로 예측해서 맞추거나, 높은  확률로 예측해서 틀리는 경우 loss가 더 크다.
metrics='accuracy' : accuracy는 정확도를 판단해주는 지표이다.

model.fit에서 0,1 예측 프로그램을 만들기 위해선 output을 0,1로만 한정시켜야 한다. 
따라서 input에서 들어온 값을 매 Activation을 시켜 결과값을 한정시킨다( "y = sigmoid(wx+b)" ).
'''

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=300, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])




#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
# result = model.predict(x_test)
# print(result)

import matplotlib.pyplot as plt


'''
loss :  [0.23306547105312347, 0.9122806787490845]
'''
