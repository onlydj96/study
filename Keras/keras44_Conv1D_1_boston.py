
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston 
import time

#1. 데이터 구성
datasets = load_boston()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66)
#print(x_train.shape, x_test.shape) # (354, 13) (152, 13)

# CNN 데이터 형식에 맞춰 변환
x_train = x_train.reshape(354, 13, 1)
x_test = x_test.reshape(152, 13, 1)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(300, 2, input_shape=(x.shape[1], 1)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.2)
end = time.time() - start
print("걸린 시간 : ", round(end, 3))

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어", r2)

'''
LSTM
걸린 시간 :  17.496
loss :  12.326574325561523
r2스코어 0.8507987486343058

Conv1D
걸린 시간 :  4.455
loss :  46.698936462402344
r2스코어 0.43475460953679934
'''