
# ReduceLRPleatu 실습

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import MinMaxScaler


#1. 데이터 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

# x에 대한 전처리
scaler = MinMaxScaler() 
x_train = x_train.reshape(50000, -1)  
x_test = x_test.reshape(10000, -1)
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 28, 28, 3)  
x_test = x_test.reshape(10000, 28, 28, 3)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), padding='valid', activation='relu', input_shape=(28, 28, 1))) 
model.add(Dropout(0.2))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))   
model.add(MaxPool2D())

model.add(Conv2D(64, (2, 2), activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# model.summary()


#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr])


#4. 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)



'''
loss :  [0.7802318930625916, 0.7346000075340271]
'''