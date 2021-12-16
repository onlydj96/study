
# 모델구성을 자유자재로 다룰 수 있게끔 연습
# summary를 통해 연산값과 차원의 배치를 확인

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv1D, Flatten, Dropout, MaxPool1D, MaxPool2D, Reshape, LSTM


#1. 데이터 전처리 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# x에 대한 전처리
x_train = x_train.reshape(60000, 28, 28, 1)  
x_test = x_test.reshape(10000, 28, 28, 1)

# y에 대한 전처리(원핫인코딩)'

# print(np.unique(y_train, return_count=True)) # [0 1 2 3 4 5 6 7 8 9]
# return_count=True 함수는 전체 개수에서 np.unique의 각 컬럼의 개수가 나옴 

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), padding='same', input_shape=(28, 28, 1)))   # 28, 28, 6
model.add(MaxPool2D())
model.add(Conv2D(5, (2, 2), activation='relu'))   # (N, 13, 13, 5)
model.add(Conv2D(7, (2, 2), activation='relu'))   # (N, 12, 12, 7)
model.add(Conv2D(7, (2, 2), activation='relu'))   # (N, 11, 11, 7)
model.add(Conv2D(10, (2, 2), activation='relu'))  # (N, 10, 10, 10)

model.add(Flatten())                              # (N, 1000)
model.add(Reshape(target_shape=(100, 10)))        # (N, 100, 10)

model.add(Conv1D(5, 2))                           # (N, 99, 5)
model.add(LSTM(15))                               # (N, 15)  
model.add(Dense(10, activation='softmax'))

model.summary()

# #3. 컴파일
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
# model.fit(x_train, y_train, epochs=300, batch_size=32, validation_split=0.2)

# #4. 예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# y_pred = model.predict(x_test)