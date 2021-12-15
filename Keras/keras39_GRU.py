
# GRU 모델의 기능과 summary

'''
GRU는 LSTM보다 gate가 하나 적기 때문에 연산량이 적고, 연산속도가 더 빨라진다. 하지만 성능이 좋아지는 것은 아니다.
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU

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
model.add(GRU(10, input_shape=(3, 1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()
