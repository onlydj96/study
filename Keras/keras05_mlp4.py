import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10)])
x = x.T

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
             [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
y = y.T

#2. 모델 구성
model = Sequential()
model.add(Dense(500, input_dim=1))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer ='adam')
model.fit(x, y, epochs=500, batch_size = 1)

#4. 평가, 예측 
loss = model.evaluate(x, y)
print('loss : ', loss)
y_predict = model.predict([9])
print('[10, 1.3, 1]의 예측값 : ', y_predict)

'''
loss :  0.008362168446183205
[10, 1.3, 1]의 예측값 :  [[9.999907   1.526834   0.93274367]]
'''