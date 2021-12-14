
#  Epoch의 기능과 결과값

# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3, 5, 4])
y = np.array([1, 2, 3, 4, 5])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)
result = model.predict([6])
print('6의 예측값 : ', result)


'''
epochs = 4000 -
loss : 0.38005509972572327

5000
loss : 0.3800192

6000 -
loss : 0.38000401854515076

15000 -
loss : 0.38000017404556274

30000 -
loss : 0.3800000548362732
'''