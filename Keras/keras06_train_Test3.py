
# train과 test를 분리하는 함수적용

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

# 데이터셋을 무작위로 분할
from sklearn.model_selection import train_test_split  # 데이터를 랜덤을 분할해서 모델링을 할 때 필요한 함수

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66) 

'''
데이터를 분할하는 경우 split으로 순차적으로 분할하지 않고 무작위로 분할해야 더 정확한 값(the optimum loss rate)이 도출된다.

test_size= : 백분위 비율로 나뉘는 값
shuffle=True : 랜덤함수이며, True가 디폴트값이다. False의 경우 순차적으로 배열시킴
random_state= : 난수의 초기값을 설정함. 번호는 아무거나 상관없음
'''

print(x_test) # [ 8 93  4  5 52 41  0 73 88 68]
print(y_test) # [ 9 94  5  6 53 42  1 74 89 69]



model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)


loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
result = model.predict([100])
print("100의 예측값 : ", result)

'''
loss :  7.977600762387738e-05
100의 예측값 :  [[101.017845]]
'''