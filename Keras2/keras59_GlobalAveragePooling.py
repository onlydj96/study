
# GlobalAveragePooling2D 사용방법

import numpy as np
import pandas as pd
from sympy import Max
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터 전처리 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# x에 대한 전처리
x_train = x_train.reshape(60000, 28, 28, 1)  
x_test = x_test.reshape(10000, 28, 28, 1)

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

# model.add(Flatten())  
model.add(GlobalAveragePooling2D())  # Flatten을 대체하여 사용가능
model.add(Dense(10, activation='softmax'))

# model.summary()

#3. 컴파일
from tensorflow.keras.optimizers import Adam
import time

optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
end = time.time()-start
print("걸린 시간 : ", round(end, 2))

#4. 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', round(loss, 2))
print('acc : ', round(acc, 2))


'''

GlobalAveragePool2D : 0.0001
걸린 시간 :  149.93
loss :  0.09
acc :  0.97
'''