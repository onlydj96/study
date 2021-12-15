import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
datasets = fetch_covtype()
x = datasets.data # (581012, 54)
y = datasets.target # (581012, )

import pandas as pd
y = pd.get_dummies(y) # [1 2 3 4 5 6 7]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

print(x_train.shape, x_test.shape)  # (464809, 54) (116203, 54)

x_train = x_train.reshape(464809, 9, 6)
x_test = x_test.reshape(116203, 9, 6)

#2. 모델 구성
model = Sequential()
model.add(LSTM(100, input_shape=(9, 6)))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(7, activation='softmax'))

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='auto', patience=10, restore_best_weights=True)
model.fit(x_train, y_train, epochs=50, batch_size=300, validation_split=0.2) 

#4. 예측, 결과
loss = model.evaluate(x_test, y_test)
print('loss, accuracy : ', loss)

'''
DNN
loss, accuracy :  [0.647487461566925, 0.720738708972930]

RNN
loss, accuracy :  [0.3921349048614502, 0.8255897164344788]
'''