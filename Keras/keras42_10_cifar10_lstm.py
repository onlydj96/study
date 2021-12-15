from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers.recurrent import LSTM


#1. 데이터 정제
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 스케일러 적용
scaler = MinMaxScaler()
x_train = x_train.reshape(50000, -1)  # (50000, 3072)
x_test = x_test.reshape(10000, -1)  # (10000, 3072)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 원핫인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# RNN의 데이터 형태로 변환
x_train = x_train.reshape(50000, 64, 48)
x_test = x_test.reshape(10000, 64, 48)

#2. 모델구성

model=Sequential()
model.add(LSTM(64, input_shape=(64, 48)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])

#4. 예측
loss = model.evaluate(x_test, y_test)
print("loss, accuracy : ", loss)

'''
DNN
loss, accuracy :  [1.4983947277069092, 0.4722000062465668]

RNN
loss, accuracy :  [0.30976781249046326, 0.8884999752044678]
'''