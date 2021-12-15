import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston  # sklearn.datasets에서는 자체에서 제공해주는 자료가 있음

#1. 데이터 구성
datasets = load_boston()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66)

#print(x_train.shape, x_test.shape) # (354, 13) (152, 13)

x_train = x_train.reshape(354, 13, 1)
x_test = x_test.reshape(152, 13, 1)

#2. 모델 구성
model = Sequential()
model.add(LSTM(300, input_shape=(13, 1)))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어", r2)

'''
DNN
loss :  16.589523315429688
r2스코어 0.7991998955875229

RNN
loss :  12.326574325561523
r2스코어 0.8507987486343058
'''