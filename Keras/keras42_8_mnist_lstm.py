from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers.core import Dropout


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# 스케일러 적용
scaler = StandardScaler()
x_train = x_train.reshape(60000, -1)  # (60000, 784)
x_test = x_test.reshape(10000, -1)  # (10000, 784)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# y 원핫인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# RNN 모델 데이터로 변환
x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

#2. 모델구성
model=Sequential()
model.add(LSTM(64, input_shape=(28, 28)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=300, validation_split=0.2, callbacks=[es])


#4. 예측
loss = model.evaluate(x_test, y_test)
print("loss, accuracy : ", loss)


'''
DNN
loss, accuracy :  [0.16708636283874512, 0.9592999815940857]

RNN
loss, accuracy :  [0.07796159386634827, 0.9789000153541565]
'''