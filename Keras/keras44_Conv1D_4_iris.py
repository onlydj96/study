
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPool1D, Flatten
from sklearn.model_selection import train_test_split
import time


#1. 데이터
datasets = load_iris()
x = datasets.data # (150, 4)
y = datasets.target # (150,)

# data scaling
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

# data split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
print(x_train.shape, x_test.shape)  # (120, 4) (30, 4)

# # RNN 데이터 형식에 맞춰 변환
x_train = x_train.reshape(120, 4, 1)
x_test = x_test.reshape(30, 4, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(100, 2, input_shape=(4, 1)))
model.add(MaxPool1D())
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
start = time.time()
model.fit(x_train, y_train, epochs=500, validation_split=0.2)
end = time.time() - start
print("걸린 시간 : ", round(end, 3))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss, accuracy  : ", loss)


'''
LSTM
걸린 시간 :  11.892
loss, accuracy  :  [0.056159455329179764, 1.0]

Conv1D
걸린 시간 :  8.581
loss, accuracy  :  [0.16826778650283813, 0.8999999761581421]
'''