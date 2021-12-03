import numpy as np

# 데이터
x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array([range(701, 801)])

# print(x.shape, y.shape)  # (3, 100), (1, 10)
x = np.transpose(x)
y = np.transpose(y)   # x = (100, 3), y = (100, 1)

x = x.reshape(1, 10, 10, 3) # (1, 10, 10, 3)

'''
만일 x = x.reshape(1, 10, 10, 3) 와 같이 바꾼다면,
사진과 같은 4차원 출력 행렬 등 3차원 이상의 행렬은 다른 모델링을 해주어야한다.
'''

# 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(Dense(10, input_dim=3))      # (100, 3) -> (N, 3) 즉 input_dim에서 행은 무시한다.
model.add(Dense(10, input_shape=(3,)))   # input_shape는 앞의 행을 제외하여 표시한다.
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(1))
# model.summary()

