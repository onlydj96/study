
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPool1D, Flatten
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import time

#1. 데이터 정제
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# data scaling
scaler = MinMaxScaler()
x_train = x_train.reshape(50000, -1)  # (50000, 3072)
x_test = x_test.reshape(10000, -1)  # (10000, 3072)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# One Hot Encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# RNN의 데이터 형태로 변환
x_train = x_train.reshape(50000, 64, 48)
x_test = x_test.reshape(10000, 64, 48)

#2. 모델구성
model=Sequential()
model.add(Conv1D(64, 2, input_shape=(64, 48)))
model.add(MaxPool1D())
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=100, validation_split=0.2)
end = time.time()-start
print("걸린 시간 : ", round(end, 3))

#4. 예측
loss = model.evaluate(x_test, y_test)
print("loss, accuracy : ", loss)

'''
LSTM
걸린 시간 :  460.899
loss, accuracy :  [1.3652085065841675, 0.5206000208854675]

Conv1D
걸린 시간 :  72.228
loss, accuracy :  [2.0862221717834473, 0.4643000066280365]
'''