
# Conv1D 모델의 기능과 모델 구현

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional, Conv1D, Flatten

#1. 데이터
x = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5],
              [4, 5, 6]])

y = np.array([4, 5, 6, 7])

print(x.shape, y.shape)      # (4, 3) (4,) 

x = x.reshape(4, 3, 1) 

#2. 모델구성
model = Sequential()
model.add(Conv1D(10, 2, input_shape=(3, 1)))
model.add(Dense(10, activation='relu'))
model.add(Flatten())  # Flatten해서 출력할 경우는 2D로 Conv1D 그대로 출력하면 3D로 나온다.
model.add(Dense(1))

model.summary()
'''
Conv1D는 3차원 배열구조를 가지고 있다. 주로 시계열 데이터나, 자연어처리(NLP)에서 주로 쓰인다.

Conv1D의 입력과 출력
입력 : (batch_size, timesteps, channels)
출력 : (batch_size, timesteps, filters)
'''

# #3. 컴파일 
model.compile(loss='mae', optimizer='adam')
import time
start = time.time()
model.fit(x, y, epochs=100, batch_size=1)
end = time.time() - start
print("걸린 시간 : ", round(end, 1))

# #4. 평가예측
loss = model.evaluate(x, y)
print("loss : ", loss)
result = model.predict([[[5], [6], [7]]])
result = result.round(0).astype(int).reshape(-1,)
print(result)