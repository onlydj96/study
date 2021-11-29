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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1004)

model = Sequential()
model.add(Dense(500, input_dim=10))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))



model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1500, batch_size=15, verbose=2 )

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)


y_predict = model.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("r2스코어", r2)

'''
loss :  3160.733154296875
r2스코어 0.5489939285103101
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

datasets = load_boston()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=1500, batch_size=15, verbose=1,
          validation_split=0.3)

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
result = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, result)
print("r2 스코어 : ", r2)

plt.figure(figsize=(9,5)) # 그래프의 사이즈
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')  
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
'''
plot : 그래프 위에 저장되는 데이터를 표현하는 함수
marker='.' : 데이터를 표시하는 수단을 점으로 표현
c= : color 
'''
plt.grid()    # 모눈종이
plt.title('loss') # 제목
plt.ylabel('loss') # y축의 제목(title)
plt.xlabel('epochs') # x축의 제목(title)
plt.legend(loc='upper right') # 기호설명표, loc = location
plt.show() # 그래프 시각화

'''
loss :  28.241247177124023
r2 스코어 :  0.6654623217377125
'''