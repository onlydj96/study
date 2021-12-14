
# LSTM의 parameter 연산법

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.python.keras.layers.core import Dropout

#1. 데이터
x = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5],
              [4, 5, 6]])

y = np.array([4, 5, 6, 7])

# print(x.shape, y.shape)      # (4, 3) (4,) 

x = x.reshape(4, 3, 1) 

#2. 모델구성
model = Sequential()
model.add(LSTM(10, input_shape=(3, 1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

'''
LSTM은 RNN의 특별한 한 종류로, 긴 의존 기간을 필요로 하는 학습을 수행할 능력을 갖고 있다.
'''
# nods * (nods + bias + feature)

# 4 * {10 * (10 + 1 + 1)} = 120 