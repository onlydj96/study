import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.engine.sequential import Sequential
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# print(datasets.DESCR)
# print(datasets.feature_names)   
# print(x.shape, y.shape) # (569. 30), (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

print(x_train.shape, x_test.shape) # (455, 30) (114, 30)

x_train = x_train.reshape(455, 30, 1)
x_test = x_test.reshape(114, 30, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(100, input_shape=(30, 1)))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
hist = model.fit(x_train, y_train, epochs=100, validation_split=0.2)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
result = model.predict(x_test)

'''
DNN
loss :  [0.23306547105312347, 0.9122806787490845]

RNN
loss :  [0.19827504456043243, 0.9473684430122375]
'''
