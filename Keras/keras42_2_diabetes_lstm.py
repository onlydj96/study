import numpy as np
from numpy.lib.npyio import load
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data # (442, 10)
y = datasets.target.reshape(-1, 1).astype(int) # (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

# print(x_train.shape, x_test.shape) # (353, 10) (89, 10)

x_train = x_train.reshape(353, 10, 1)
x_test = x_test.reshape(89, 10, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(300, input_shape=(10, 1)))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, validation_split=0.2)

#4. 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)

'''
DNN
loss :  3160.733154296875
r2스코어 0.5489939285103101

RNN
loss :  5092.20556640625
r2 스코어 : 0.21538151250912974
'''
