# R2 : 회귀모델이 주어진 자료에 얼마나 적합한지를 평가하는 지표
# y의 변동량대비 모델 예측값의 변동량을 의미함
# 0~1의 값을 가지며, 상관관계가 높을수록 1에 가까워짐
# r2=0.3인 경우 약 30% 정도의 설명력을 가진다 라고 해석할 수 있음
# sklearn의 r2_score의 경우 데이터가 arbitrarily할 경우 음수가 나올수 있음, 음수가 나올경우 모두 일괄 평균으로 예측하는 것보다 모델의 성능이 떨어진다는 의미
# 결정계수는 독립변수가 많아질 수록 값이 커지기때문에, 독립변수가 2개 이상일 경우 조정된 결정계수를 사용해야 함

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66)



#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=2)

#4. 평가 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)

print("r2스코어", r2)


# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show


'''
loss :  0.38005200028419495
r2스코어 0.8099739854117104
'''