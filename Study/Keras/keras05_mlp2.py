import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])


x = x.T

'''
데이터 프레임에서 행과 열을 바꾸기 위해서 필요한 함수
df.transpose()
df.T

* 위 두개의 함수와 df.reshape()은 동일하게 행과 열을 바꾸는 함수이지만 
df.transpose(), df.T는 행과 열을 하나씩 차례대로 바꾸고 df.reshape()은 행이 다 바뀌어지면 그 다음 행이 바뀌는 식의 구조이다.
따라서 대부분의 경우는 reshape()을 쓴다.
'''

#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=3))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
y_predict = model.predict([[10, 1.3, 1]])
print('[10, 1.3, 1]의 예측값 : ', y_predict)


'''
loss :  5.470137764973515e-08
[10, 1.3, 1]의 예측값 :  [[20.000326]]
'''