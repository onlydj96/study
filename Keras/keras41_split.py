
# RNN 구조에서 데이터를 split하는 함수 구현

import numpy as np

a = np.array(range(1, 101))  
x_predict = np.array(range(96, 106))
size = 5

def split_x(dataset, size):
    list = []
    for i in range(len(dataset) - size + 1):  # i : range 만큼 반복될 횟수를 하나씩 순서대로 반환하는 변수
        subset = dataset[i : (i + size)]
        list.append(subset)
    return np.array(list)

dataset = split_x(a, size)

# print(dataset.shape) # (96, 5)

x = dataset[:, :-1]  
y = dataset[:, -1]
print(x.shape, y.shape)  # (96, 4) (96,)





