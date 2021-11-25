# R2를 음수가 아닌 0.5 이하로 만들것 
# 데이터는 건들지 말것
# 레이어는 인풋 아웃풋 포함 6개 이상
# batch_size = 1
# epochs = over 100
# each nod of hidden layer = over 10 and less 1000
# train = 70%

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66)



#2. 모델 구성
model = Sequential()
model.add(Dense(444, input_dim=1))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print("r2스코어", r2)



