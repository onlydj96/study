from tensorflow.keras.datasets import cifar10
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler



#1. 데이터 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

# scaling 데이터 reshape
scaler = MinMaxScaler() 
x_train = x_train.reshape(50000, -1)  # scaler를 활용하기 위해서는 2차원 데이터로 변환해야함
x_test = x_test.reshape(10000, -1)
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 이미지 데이터 reshape
x_train = x_train.reshape(50000, 32, 32, 3)  
x_test = x_test.reshape(10000, 32, 32, 3)

# # y에 대한 전처리(원핫인코딩)
# print(np.unique(y_train, return_counts=True)) # [0 1 2 3 4 5 6 7 8 9]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(30, kernel_size=(2,2), strides=2, padding='same', input_shape=(32, 32, 3)))   
model.add(MaxPool2D())
model.add(Conv2D(10, (2,2), activation='relu')) 
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(30))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))
# model.summary()

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.3, callbacks=[es])


#4. 예측
loss = model.evaluate(x_test, y_test)
print("loss, accuracy : ", loss)

acc = str(round(loss[1], 4))
model.save("./_save/cifar_{}.h5".format(acc))


'''
loss, accuracy :  [1.3061442375183105, 0.5454000234603882]
'''

