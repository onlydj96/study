
# train과 test 분리를 비율에 맞춰서 하는 방법

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# train과 test를 8:2으로 분류하시오.
x_train = x[:8]
x_test = x[8:]
y_train = y[:8]
y_test = y[8:]

#2. 모델 구현
model = Sequential()
model.add(Dense(100, input_dim = 1))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=3000, batch_size = 1)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('11의 예측값 : ', result)

'''
loss :  5.9117155615240335e-12
11의 예측값 :  [[10.999999]]
'''