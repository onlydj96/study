
# ReduceLR 사용 방법

from ast import Global
import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAveragePooling2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


#1. 데이터 전처리 
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (50000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# x에 대한 전처리
x_train = x_train.reshape(50000, 32, 32, 3) / 255
x_test = x_test.reshape(10000, 32, 32, 3) / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='valid', activation='relu', input_shape=(32, 32, 3))) 
model.add(Dropout(0.2))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))   
model.add(MaxPool2D())

model.add(Conv2D(64, (2, 2), activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(MaxPool2D())

# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='softmax'))

# model.summary()

#3. 컴파일
from tensorflow.keras.optimizers import Adam
import time

optimizer = Adam(learning_rate=0.00001)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=15)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es, reduce_lr])
end = time.time()-start
print("걸린 시간 : ", round(end, 2))

#4. 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', round(loss, 2))
print('acc : ', round(acc, 2))

'''
걸린 시간 :  1064.38
loss :  2.18
acc :  0.44
'''