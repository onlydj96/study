
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

x = dataset[:, :-1]
y = dataset[:, -1]

print(a.shape, x.shape, y.shape)
# 예측 데이터 구성
x = x.reshape(96, 4, 1)
dataset2 = split_x(x_predict, size)
x_pred = dataset2[:, :-1].reshape(6, 4, 1)