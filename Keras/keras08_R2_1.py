
# R2 스코어 적용

'''
R2 : 회귀모델이 주어진 자료에 얼마나 적합한지를 평가하는 지표이다. y의 변동량대비 모델 예측값의 변동량을 의미함
r2 score는 0~1의 값을 가지며, 상관관계가 높을수록 1에 가까워진다. 즉 r2=0.3인 경우 약 30% 정도의 설명력을 가진다 라고 해석할 수 있다
sklearn의 r2_score의 경우 데이터가 arbitrarily할 경우 음수가 나올수 있다, 음수가 나올경우 모두 일괄 평균으로 예측하는 것보다 모델의 성능이 떨어진다는 것을 의미한다
결정계수는 독립변수(x)가 많아질 수록 값이 커지기때문에, 독립변수가 2개 이상일 경우 조정된 결정계수를 사용해야 함
'''

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([1, 2, 4, 3, 5, 7, 9, 9, 8, 12, 13, 17, 12, 14, 21, 14, 11, 19, 23, 25])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66)



#2. 모델 구성
model = Sequential()
model.add(Dense(500, input_dim=1))
model.add(Dense(200))
model.add(Dense(70))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=1)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score  # r2 함수
r2 = r2_score(y_test, y_predict)

print("r2스코어", r2)


# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show


'''
loss :  6.161921977996826
r2스코어 0.5667398608640118
'''