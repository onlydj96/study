import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split


#1. 데이터
datasets = load_iris()
x = datasets.data # (150, 4)
y = datasets.target # (150,)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
print(x_train.shape, x_test.shape)  # (120, 4) (30, 4)

x_train = x_train.reshape(120, 4, 1)
x_test = x_test.reshape(30, 4, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(100, input_shape=(4, 1)))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3, activation='softmax'))  

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.fit(x_train, y_train, epochs=500, validation_split=0.2)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss, accuracy  : ", loss)


'''
DNN
loss, accuracy :  [0.07134769856929779, 1.0]

RNN
loss, accuracy  :  [0.056718528270721436, 1.0]
'''