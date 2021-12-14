
# CNN의 기본적인 모델구성 및 부속 함수용어 설명
# CNN의 parameter연산 원리

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.backend import conv2d
from tensorflow.python.keras.layers.core import Dropout



model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), strides=1, padding='same', input_shape=(10, 10, 1)))   # 9, 9, 10

# ( kernel_size + bias ) + filter = (2x2x1+1)x10 = 50

model.add(MaxPool2D())
model.add(Conv2D(5, (2, 2), activation='relu'))   # 7, 7, 5
model.add(Conv2D(7, (2, 2), activation='relu'))   # 6, 6, 7
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(5, activation='softmax'))

# model.summary()


'''
1. Conv2D : 확산곱 함수, 여기에 개수는 filter이다.
2. kernel_size : 이미지 데이터의 행렬을 나누는 단위. ex) (10,10)을 (2,2)씩 나뉘면 총 (5,5)의 행렬이 나온다. 
kernel_size를 통해 이미지를 나누어 계산하며 데이터의 특성을 찾고 머신이 이미지를 인식할 수 있게 한다.
3. strides=n : kernel의 행렬을 n만큼 이동하여 계산한다.
4. padding='same' : shape이 아래 layer로 내려갈 때 값을 그대로 고정시키는 함수
5. input_shape=() : 2차원 이상의 input값을 다룰 때의 사용하는 함수. 위는 이미지 데이터(4차원) 이기 때문에 (행, 렬, 컬러)로 나누었다
6. maxpooling : kernel_size를 해서 가장 수치가 높게 나온 데이터만 정제하여 과적합을 방지하는 함수. (model에 dropout함수와 비슷한 개념)
7. Flatten : 3차원의 데이터를 1차원 array로 바꾸어 Dense 모델링을 할 수 있게 만드는 함수. 열 단위로 나뉜 행들을 첫번째 행에 순서대로 붙여 늘린다.
'''