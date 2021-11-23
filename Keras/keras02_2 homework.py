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
model.add(Dense(500))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=50, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)
result = model.predict([4])
print('4의 예측값 : ', result)