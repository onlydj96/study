
# CNN에 Scaler 적용하는 방법

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import MinMaxScaler


#1. 데이터 전처리
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()  # 이미지 데이터 불러오기, train, test구분

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# x에 대한 전처리
scaler = MinMaxScaler() 
x_train = x_train.reshape(60000, -1)  # scaler를 활용하기 위해서는 2차원 데이터로 변환해야함
x_test = x_test.reshape(10000, -1)
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)  
x_test = x_test.reshape(10000, 28, 28, 1)

# y에 대한 전처리(원핫인코딩)
# print(np.unique(y_train, return_counts=True))
# return_count=True 함수는 전체 개수에서 np.unique의 각 컬럼의 개수가 나옴 

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(3,3), strides=2, padding='same', input_shape=(28, 28, 1)))   # 27, 27, 6  
model.add(MaxPool2D())
model.add(Conv2D(5, (3,3), activation='relu'))   # 7, 7, 5
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

# model.summary()


#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True)
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])


#4. 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)

acc = str(round(loss[1], 4))
model.save("./_save/fashion_{}.h5".format(acc))


'''
loss :  [0.3363991975784302, 0.8810999989509583]
'''