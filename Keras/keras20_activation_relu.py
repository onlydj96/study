# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델구성
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(500, activation='relu')) # 'relu'함수는 layer에서 layer로 넘어갈 때 음수값이 나오는 nod는 0처리해준다.
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)
result = model.predict([4])
print('4의 예측값 : ', result)


'''
loss : 0.0
4의 예측값 :  [[4.]]
'''
