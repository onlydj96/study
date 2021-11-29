import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
import time

datasets = load_diabetes()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

model = Sequential()
model.add(Dense(500, input_dim=10))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)


start = time.time()
hist = model.fit(x_train, y_train, epochs=10000, batch_size=1, verbose=1,
          validation_split=0.2, callbacks=[es])  # model.fit에 EarlyStopping을 적용
end = time.time() - start
print("걸린 시간 : ", round(end, 2), '초')

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
result = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, result)
print("r2 스코어 : ", r2)


'''
Earlystopping은 minimum val_loss 값을 찾고나서 바로 멈추지 않고 patience= 만큼 더 최저값을 찾는 과정을 거친다. 
문제는 Earlystopping으로 멈춘 model.fit은 Earlystopping point가 아닌 patience만큼 지난 값을 가중치로 갖는다는 것이다.
따라서 restore_best_weights = '라는 함수가 early stopping한 최적의 loss값을 복원하여 test값을 출력한다. 
'''