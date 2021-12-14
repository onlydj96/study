
# input_shape을 다른 방식으로 구현

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터
x = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5],
              [4, 5, 6]])

y = np.array([4, 5, 6, 7])

print(x.shape, y.shape)      # (4, 3) (4,) 

y = y.reshape(-1, 1)
x = x.reshape(4, 3, 1)  # batch_size = 4, timesteps = 3, feature = 1

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(3, 1), activation='linear'))
model.add(SimpleRNN(10, input_length=3, input_dim=1))  # input_shape(3, 1)를 다른 방식으로 구현
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
