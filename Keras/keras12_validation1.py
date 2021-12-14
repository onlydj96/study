
# validation의 의미와 적용법

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x_train = np.array(range(1, 11))
y_train = np.array(range(1, 11))
x_test = np.array([11, 12, 13])
y_test = np.array([11, 12, 13])
x_val = np.array([14, 15, 16])
y_val = np.array([14, 15, 16])


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1, verbose=1, 
          validation_data=(x_val, y_val))  

'''
validation 이란 test에서 평가하기 전에 train 단계에서 자체적으로 평가하는 것을 의미한다. 
'''

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)

'''
loss :  6.063298192519884e-13
17의 예측값 :  [[17.000002]]
'''