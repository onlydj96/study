
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

print(x_train.shape, x_test.shape)  # (142, 13) (36, 13)

x_train = x_train.reshape(142, 13, 1)
x_test = x_test.reshape(36, 13, 1)

#2 모델구성
model = Sequential()
model.add(LSTM(100, input_shape=(13, 1)))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3, activation='softmax'))

#3 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=500, validation_split=0.2)

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss, accuracy : ", loss)


'''
DNN
loss, accuracy  : [0.18074281513690948, 0.9166666865348816]

RNN
loss, accuracy :  [0.1247176080942154, 0.9722222089767456]
'''