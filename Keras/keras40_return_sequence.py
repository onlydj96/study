
# LSTM을 연속해서 모델링을 하는 방법

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

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
model.add(LSTM(532, input_shape=(3, 1), return_sequences=True)) 
model.add(LSTM(163, return_sequences=True))
model.add(LSTM(163, return_sequences=True))
model.add(LSTM(163, return_sequences=True))
model.add(LSTM(163, return_sequences=False))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(8))
model.add(Dense(1))

'''
return_sequences=True : RNN구조의 layer를 연속으로 쌓기위해서 필요한 함수.
RNN의 layer를 통과한 데이터의 shape는 3차원에서 2차원으로 변환되기 때문에 return_sequences는 차원을 유지해준다. 

* 하지만 RNN 계열의 모델링은 연속을 쌓았을 때 성능이 좋아지지 않는다.
'''

#3. 컴파일
model.compile(loss='mae', optimizer='adam') 
model.fit(x, y, epochs=500, batch_size=2)

#4. 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

y_predict = model.predict([[[50], [60], [70]]])
result = round(float(y_predict.reshape(-1,)), 2)

print(result)
