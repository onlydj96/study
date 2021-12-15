
# LSTM 구현 방법 및 정확도

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

# model.add(SimpleRNN(10, input_shape=(3, 1)))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1))

'''
LSTM은 hidden state만이 아니라 cell state라는 역할이 있으며 Forget gate, Input gate, Output gate를 통해 계산이 이루어진다
즉, 4개의 gate를 통하여 연산하기 때문에 RNN에 비해서 4배의 많은 parameter를 가진다.

'''

#3. 컴파일 
model.compile(loss='mae', optimizer='adam') 
model.fit(x, y, epochs=1000, batch_size=1)


# #4. 평가예측
loss = model.evaluate(x, y)
print("loss : ", loss)
result = model.predict([[[5], [6], [7]]])
# result = result.round(0).astype(int).reshape(-1,)
print(result)

'''
SimpleRNN
loss :  0.019789695739746094
[[8.012796]]

LSTM
loss :  0.0915137529373169
[[7.812167]]
'''
