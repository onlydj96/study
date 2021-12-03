import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn import datasets
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MaxAbsScaler
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

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
input1 = Input(shape=(54,))
hidden1 = Dense(100)(input1)
hidden2 = Dense(60, activation='relu')(hidden1)
hidden3 = Dense(30, activation='relu')(hidden2)
hidden4 = Dense(15, activation='relu')(hidden3)
output1 = Dense(7, activation='softmax')(hidden4)
model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# model.add(Dense(100, input_dim=54))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(15, activation='relu'))
# model.add(Dense(7, activation='softmax'))

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

