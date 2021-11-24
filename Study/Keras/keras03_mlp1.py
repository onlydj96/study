import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]])
x = x.T
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])



#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=2))
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
y_predict = model.predict([[10, 1.3]])
print('[10, 1.3]의 예측값 : ', y_predict)