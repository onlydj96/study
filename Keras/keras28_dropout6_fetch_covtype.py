import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn import datasets
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.metrics import accuracy
import time

#1. 데이터
datasets = fetch_covtype()
x = datasets.data # (581012, 54)
y = datasets.target # (581012, )

import pandas as pd
y = pd.get_dummies(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=54))
model.add(Dropout(0.2))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(15, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일
from tensorflow.keras.callbacks import EarlyStopping

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") 

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'      
model_path = "".join([filepath, '6_fetch_covtype_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, filepath=model_path)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es])


#4. 예측, 결과
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
loss :  0.31304067373275757
accuracy :  0.8719568252563477
'''