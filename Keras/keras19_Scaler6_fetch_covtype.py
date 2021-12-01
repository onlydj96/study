import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn import datasets
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.python.keras.metrics import accuracy
import time

#1. 데이터
datasets = fetch_covtype()
x = datasets.data # (581012, 54)
y = datasets.target # (581012, )

import pandas as pd
y = pd.get_dummies(y)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x에 대한 전처리 완료. y값은 전처리할 필요가 없음


#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=54))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es])
end = time.time() - start
print("걸린 시간 : ", round(end, 2), "초")

#4. 예측, 결과
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


'''
- 결과값 - 
# 그냥
loss :  0.6518669128417969
accuracy :  0.7172620296478271
# MinMax
loss :  0.6326460838317871
accuracy :  0.723285973072052
# Standard
loss :  0.6319209933280945
accuracy :  0.7207129001617432
# Robuster
loss :  0.632551372051239
accuracy :  0.7227007746696472
# MaxAbs
lloss :  0.6325117945671082
accuracy :  0.7231740951538086

- relu 사용한 결과값 - 
# 그냥
loss :  1.204178810119629
accuracy :  0.4892730712890625
# MinMax
loss :  0.3027080297470093
accuracy :  0.8789790272712708
# Standard
loss :  0.2828752100467682
accuracy :  0.8853471875190735
# Robuster
loss :  0.2740715742111206
accuracy :  0.8899942636489868
# MaxAbs
loss :  [0.3029820919036865, 0.9359999895095825]
'''