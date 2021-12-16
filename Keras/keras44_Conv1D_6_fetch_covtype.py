
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPool1D, Flatten
from sklearn.model_selection import train_test_split
import time

from tensorflow.python.ops.gen_math_ops import Max

#1. 데이터
datasets = fetch_covtype()
x = datasets.data # (581012, 54)
y = datasets.target # (581012, )

# data scaling
import pandas as pd
y = pd.get_dummies(y) # [1 2 3 4 5 6 7]

# data split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)
print(x_train.shape, x_test.shape)  # (464809, 54) (116203, 54)

# # RNN 데이터 형식에 맞춰 변환
x_train = x_train.reshape(464809, 9, 6)
x_test = x_test.reshape(116203, 9, 6)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(100, 2, input_shape=(9, 6)))
model.add(MaxPool1D())
model.add(Flatten())
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(7, activation='softmax'))

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
model.fit(x_train, y_train, epochs=30, batch_size=300, validation_split=0.2) 
end = time.time() - start
print("걸린 시간 : ", round(end, 3))

#4. 예측, 결과
loss = model.evaluate(x_test, y_test)
print('loss, accuracy : ', loss)

'''
LSTM
걸린 시간 :  491.587
loss, accuracy :  [0.40976256132125854, 0.816235363483429]

Conv1D
걸린 시간 :  66.127
loss, accuracy :  [0.5907247066497803, 0.7393870949745178]
'''