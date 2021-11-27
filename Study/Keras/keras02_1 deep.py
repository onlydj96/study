# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델구성
model = Sequential()
model.add(Dense(3000, input_dim=1))
model.add(Dense(100))
model.add(Dense(5))
model.add(Dense(80))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1)) 

'''
다중 신경망을 구성히여 모델구성을 할 경우 이러한 형태를 "딥러닝" 이라고 한다. 
딥러닝(deep learning)이라 부르는 이유는 여러 개의 layer로 신경망이 순차적으로 내려가며 깊어(deep)지기 때문이다.
'''

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=50, batch_size=1)  # model.fit 에서 구성한 모델을 머신에게 훈련시킨다.

'''
epochs = 훈련양
batch_size = 한번의 layer를 거칠때 훈련하는 데이터의 개수
'''


#4. 평가, 예측
loss = model.evaluate(x, y)  # model.evalaute 에서 구성한 모델을 평가한다.
print('loss :', loss) 
result = model.predict([4])
print('4의 예측값 : ', result)

'''
loss : 1.308856667492364e-08
4의 예측값 :  [[4.0001044]]
'''