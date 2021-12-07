import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import ModelCheckpoint


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# print(datasets.DESCR) # x=(150, 4), y=(150,)
# print(np.unique(y)) # [0, 1, 2]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=4))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(5))
model.add(Dense(3, activation='softmax')) 


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
from tensorflow.keras.callbacks import EarlyStopping

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") 

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'      
model_path = "".join([filepath, '4_iris_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, filepath=model_path)

hist = model.fit(x_train, y_train, epochs=1000, verbose=1, validation_split=0.2, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss[0])
print("accuracy : ", loss[1])

'''
loss :  0.09086837619543076
accuracy :  0.9333333373069763
'''