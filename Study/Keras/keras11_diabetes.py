# R2 = 0.62이상

import numpy as np
from numpy.lib.npyio import load
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data # (442, 10)
y = datasets.target # (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=66)

model = Sequential()
model.add(Dense(500, input_dim=10))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1, verbose=2)

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)


y_predict = model.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("r2스코어", r2)

'''
loss :  2997.267822265625
r2스코어 0.518926227425742
'''
