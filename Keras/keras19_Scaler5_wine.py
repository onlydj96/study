########################################
# 각각의 Scaler의 특성과 정의를 정리하시오
########################################

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.datasets import load_wine
import time

datasets = load_wine()
x = datasets.data 
y = datasets.target

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=1004)


# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

#4. 예측, 결과
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict(x_test)




'''
- 결과값 - 
# 그냥
loss :  [5.76909875869751, 0.6000000238418579]
# MinMax
loss :  [0.11649330705404282, 0.9520000219345093]
# Standard
loss :  [0.11854667961597443, 0.9760000109672546]
# Robuster
loss :  [0.16132934391498566, 0.9520000219345093]
# MaxAbs
loss :  [0.4371863305568695, 0.8960000276565552]

- relu 사용한 결과값 - 
# 그냥
loss :  [1.104198694229126, 0.37599998712539673]
# MinMax
loss :  [0.13849852979183197, 0.9520000219345093]
# Standard
loss :  [0.2654374837875366, 0.9679999947547913]
# Robuster
loss :  [0.14828090369701385, 0.9520000219345093]
# MaxAbs
loss :  [0.3029820919036865, 0.9359999895095825]
'''