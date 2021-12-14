
# RNN의 기본적인 함수와 모델링 구현

'''
RNN : Recurrent Nerual Network의 약자로 순환 신경망이라고 불린다.
순환신경망은 특정 layer 에서 다음 layer로 가기전에 nod가 그 layer를 순환하여 학습한다.
'''

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터
x = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5],
              [4, 5, 6]])

y = np.array([4, 5, 6, 7])

print(x.shape, y.shape)      # (4, 3) (4,) 

'''
timesteps : 전체 길이중 RNN을 하기 위해 자르는 길이   ex) [1 2 3 4 5 6 7]을 [1 2 3], [2 3 4 ] 등 3개씩 자른다.
feature : timestep에서 연산하는 보폭?   ex) [1 2 3] = 1에서 2로, 2에서 3으로 한칸씩 연산
RNN에서의 input shape : (행, 열, 몇개씩 자르는지) = (batch_size, timesteps, feature) 즉, 3차원
'''

x = x.reshape(4, 3, 1)  # batch_size = 4, timesteps = 3, feature = 1

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(10, input_shape=(3, 1), activation='linear'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

'''
RNN의 모델은 3차원으로 시작하지만 끝나는 건 2차원이기때문에 Dense모델로 바로 출력할 수 있다.
ex) [[1], [2]. [3]] => RNN layer => [[4]] 
'''

#3. 컴파일 
model.compile(loss='mae', optimizer='adam') # 평가지표는 'mse', 'mae' 둘다 상관없음
model.fit(x, y, epochs=100, batch_size=1)


# #4. 평가예측
loss = model.evaluate(x, y)
print("loss : ", loss)
result = model.predict([[[5], [6], [7]]])
result = result.round(0).astype(int).reshape(-1,)
print(result)