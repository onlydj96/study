
# RNN 구조에서 데이터를 split하는 함수 구현

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))
size = 5

def split_x(dataset, size):
    list = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        list.append(subset)
    return np.array(list)

dataset = split_x(a, size)
# print(dataset)

x = dataset[:, :-1].reshape(96, 4, 1)
y = dataset[:, -1]

# 예측 데이터 구성
dataset2 = split_x(x_predict, size)
x_pred = dataset2[:, :-1].reshape(6, 4, 1)

#2. 모델
model = Sequential()
model.add(LSTM(532, input_shape=(4, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64))
model.add(Dense(16, activation='relu'))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일
model.compile(loss='mae', optimizer='adam') 
model.fit(x, y, epochs=10, batch_size=16)

#4. 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
result = model.predict(x_pred)

print(result)


'''
loss :  0.4722616374492645
[[ 98.78775 ]
 [ 99.68584 ]
 [100.571014]
 [101.44252 ]
 [102.29962 ]
 [103.1418  ]]
'''
