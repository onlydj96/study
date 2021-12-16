
import numpy as np
from numpy.lib.npyio import load
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPool1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data # (442, 10)
y = datasets.target.reshape(-1, 1).astype(int) # (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
# print(x_train.shape, x_test.shape) # (353, 10) (89, 10)

# CNN 데이터 형식에 맞춰 변환
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
start = time.time()
model.fit(x_train, y_train, epochs=200, validation_split=0.2)
end = time.time()-start
print("걸린시간 : ", round(end, 3))

#4. 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)

'''

RNN
걸린시간 :  27.339
loss :  5343.9033203125
r2 스코어 :  0.17659931821778418

CNN
걸린시간 :  6.711
loss :  3537.06884765625
r2 스코어 :  0.45500050721816676
'''
