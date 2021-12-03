import numpy as np

# 데이터
x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array([range(701, 801)])

# print(x.shape, y.shape)  # (3, 100), (1, 10)
x = np.transpose(x)
y = np.transpose(y)   # x = (100, 3), y = (100, 1)

x = x.reshape(1, 10, 10, 3) # (1, 10, 10, 3)

# 모델
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input

# 함수형 모델 구성
input1 = Input(shape=(3,))
hidden1 = Dense(10)(input1)
hidden2 = Dense(9)(hidden1)
hidden3 = Dense(8)(hidden2)
output1 = Dense(1)(hidden3)
model = Model(inputs=input1, outputs=output1)

'''
Model 함수를 쓰기위해서 Model과 Input을 inport 해야한다.
Sequential과 Model의 모델링의 결과값은 동일하다. 
Model 함수로 모델구성을 할경우, summary에 input layer를 추가적으로 표기한다.
'''

# model = Sequential()
# # model.add(Dense(10, input_dim=3))    
# model.add(Dense(10, input_shape=(3,))) 
# model.add(Dense(9))
# model.add(Dense(8))
# model.add(Dense(1))
model.summary()

