
# 저장된 모델을 불러오는 법

import numpy as np
from tensorflow.keras.models import load_model # save되어있는 model을 import하는 함수
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

datasets = load_boston()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

# model = Sequential()
# model.add(Dense(100, input_dim=13))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(3))
# model.add(Dense(1))

model =load_model("./_save/keras25_1_save_model.h5")
model.summary() # 내가 load한 모델이 맞는지 확인

# 컴파일
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=1,
                validation_split=0.3)


loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
result = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, result)
print("r2 스코어 : ", r2)
