from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.backend import conv2d
from tensorflow.python.keras.layers.core import Dropout



model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(10, 10, 1)))   # 9, 9, 10
model.add(Conv2D(5, (3, 3), activation='relu'))   # 7, 7, 5
model.add(Conv2D(7, (2, 2), activation='relu'))   # 6, 6, 7
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(5, activation='softmax'))


# 3차원 데이터를 1차원으로 바꾸는 방법은? 쫘~악 펴주는것 

'''
kernel_size : 이미지 데이터의 행렬을 나누는 단위. ex) (10,10)을 (2,2)씩 나뉘면 총 (5,5)의 행렬이 나온다. 
kernel_size를 통해 이미지를 나누어 계산하며 데이터의 특성을 찾고 머신이 이미지를 인식할 수 있게 한다.

input_shape=() : 2차원 이상의 input값을 다룰 때의 사용하는 함수. 위는 이미지 데이터(4차원) 이기 때문에 (행, 렬, 컬러)로 나누었다

Flatten : 3차원의 데이터를 1차원 array로 바꾸어 Dense 모델링을 할 수 있게 만드는 함수. 열 단위로 나뉜 행들을 첫번째 행에 순서대로 붙여 늘린다.
'''