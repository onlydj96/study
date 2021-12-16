
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPool1D, Flatten
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import time

#1. 데이터 정제
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

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

# RNN 데이터 모델로 변형
x_train = x_train.reshape(50000, 64, 48)
x_test = x_test.reshape(10000, 64, 48)  

#2. 모델구성
model=Sequential()
model.add(Conv1D(64, 2, input_shape=(64, 48)))
model.add(MaxPool1D())
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=300, validation_split=0.3)
end = time.time()-start
print("걸린 시간 : ", round(end, 3))

#4. 예측
loss = model.evaluate(x_test, y_test)
print("loss, accuracy : ", loss)

'''
LSTM
걸린 시간 :  324.357
loss, accuracy :  [3.3263678550720215, 0.20720000565052032]

Conv1D
걸린 시간 :  61.834
loss, accuracy :  [3.516528844833374, 0.20200000703334808]
'''