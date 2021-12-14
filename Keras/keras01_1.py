
# Deep Learning의 기본 구성

# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=500, batch_size=1) 

'''
batch : 가중치 등의 매개 변수의 값을 조정하기 위해 사용하는 데이터의 양
1epoch당 시간은 오래 걸리고 메모리를 크게 요구하나, 전역 최솟값을 찾을 수 있다
'''

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)
result = model.predict([4])
print('4의 예측값 : ', result)


'''
loss : 4.47883536480731e-07
4의 예측값 :  [[3.9986327]]
'''