
# X와 Y값을 분리하여 train과 test 데이터를 구현

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7])
x_test = np.array([8, 9, 10])

y_train = np.array([1, 2, 3, 4, 5, 6, 7])
y_test = np.array([8, 9, 10])

'''
test와 train을 나누는 이유는 train에서 훈련하여 만든 최적의 loss값으로 test를 진행하기 위함이다. 
이렇게 분할하여 모델링을 하지 않으면 over training되어 머신은 오류값을 반환한다.
'''

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
loss :  3.031649096259942e-13
11의 예측값 :  [[10.999999]]
'''