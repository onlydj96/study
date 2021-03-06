
# 예측값을 80을 만들어라, 하지만 80은 절대 못만듦...

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM

#1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
              [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
              [9, 10, 11], [10, 11, 12], [20, 30, 40],
              [30, 40, 50], [40, 50, 60]])

y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

print(x.shape, y.shape) #(13, 3) (13,)

x = x.reshape(13, 3, 1)

#2. 모델
model = Sequential()
model.add(LSTM(532, input_shape=(3, 1)))
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일
from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', mode='min', patience=100, restore_best_weights=True)
model.compile(loss='mae', optimizer='adam') 
model.fit(x, y, epochs=500, batch_size=2)

#4. 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
y_predict = model.predict([[[50], [60], [70]]],)

result = round(float(y_predict.reshape(-1,)), 2)

print(result)

model.save("./_save/LSTM_y_predict_{}.h5".format(result))


'''
loss :  1.233656644821167
[[78.84124]]
'''