
# Bidirectional 모델의 기능과 구현

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional

#1. 데이터
x = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5],
              [4, 5, 6]])

y = np.array([4, 5, 6, 7])

print(x.shape, y.shape)      # (4, 3) (4,) 

x = x.reshape(4, 3, 1) 

#2. 모델구성
model = Sequential()
model.add(Bidirectional(SimpleRNN(10, input_shape=(3, 1)))) 
'''
Bidirectional : RNN구조의 layer를 반대로도 한번 순차연산시켜주는 함수
'''

model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# #3. 컴파일 
# model.compile(loss='mae', optimizer='adam')
# model.fit(x, y, epochs=100, batch_size=1)


# # #4. 평가예측
# loss = model.evaluate(x, y)
# print("loss : ", loss)
# result = model.predict([[[5], [6], [7]]])
# result = result.round(0).astype(int).reshape(-1,)
# print(result)