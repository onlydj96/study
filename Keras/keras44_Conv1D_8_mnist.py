
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPool1D, Flatten
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time


#1. 데이터 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# data scaling
scaler = StandardScaler()
x_train = x_train.reshape(60000, -1)  # (60000, 784)
x_test = x_test.reshape(10000, -1)  # (10000, 784)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# One Hot Encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# RNN 데이터 형식에 맞춰 변환
x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

#2. 모델구성
model=Sequential()
model.add(Conv1D(64, 2, input_shape=(28, 28)))
model.add(MaxPool1D())
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=300, validation_split=0.2)
end = time.time()-start
print("걸린 시간 : ", round(end, 3))


#4. 예측
loss = model.evaluate(x_test, y_test)
print("loss, accuracy : ", loss)


'''
LSTM
걸린 시간 :  178.739
loss, accuracy :  [0.09151565283536911, 0.9817000031471252]

Conv1D
걸린 시간 :  33.1
loss, accuracy :  [0.19618065655231476, 0.9732000231742859]
'''