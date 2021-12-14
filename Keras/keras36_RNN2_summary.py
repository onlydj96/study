
# RNN의 parameter 연산법


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

'''
timesteps : 전체 길이중 RNN을 하기 위해 자르는 길이   ex) [1 2 3 4 5 6 7]을 [1 2 3], [2 3 4 ] 등 3개씩 자른다.
feature : timestep에서 연산하는 보폭?   ex) [1 2 3] = 1에서 2로, 2에서 3으로 한칸씩 연산
RNN에서의 input shape : (행, 열, 몇개씩 자르는지) = (batch_size, timesteps, feature) 즉, 3차원
'''

x = x.reshape(4, 3, 1)  # batch_size = 4, timesteps = 3, feature = 1

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(10, input_shape=(3, 1), activation='linear'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()   # parameter = nods * (nods + feature + bias)
