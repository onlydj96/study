
# validation에서 자체적으로 split하는 방법

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8125, random_state=66)


# x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=1004)


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
model.fit(x_train, y_train, epochs=3000, batch_size=15, verbose=1,
          validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict([18])
print("18의 예측값 : ", y_predict)

'''
loss :  3.07901843127692e-13
18의 예측값 :  [[18.]]
'''

