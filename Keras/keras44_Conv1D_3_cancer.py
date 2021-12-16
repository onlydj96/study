
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPool1D, Flatten
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.engine.sequential import Sequential
from sklearn.model_selection import train_test_split
import time

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# print(datasets.DESCR)
# print(datasets.feature_names)   
# print(x.shape, y.shape) # (569. 30), (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)
print(x_train.shape, x_test.shape) # (455, 30) (114, 30)

# RNN 데이터 형식에 맞춰 변환
x_train = x_train.reshape(455, 30, 1)
x_test = x_test.reshape(114, 30, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(100, 2, input_shape=(30, 1)))
model.add(MaxPool1D())
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time() 
hist = model.fit(x_train, y_train, epochs=100, validation_split=0.2)
end = time.time() - start
print("걸린 시간 : ", round(end, 3))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
result = model.predict(x_test)

'''
LSTM
걸린 시간 :  13.205
loss :  [0.16811339557170868, 0.9561403393745422]

Conv1D
걸린 시간 :  4.224
loss :  [0.24569976329803467, 0.9122806787490845]
'''
