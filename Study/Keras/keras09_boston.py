import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston


datasets = load_boston()
x = datasets.data
y = datasets.target

'''
print(datasets.feature_names) # 데이터셋의 특성의 이름들
print(datasets.DESCR) # 데이터셋을 소개, 설명
'''


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=66)



#2. 모델 구성
model = Sequential()
model.add(Dense(500, input_dim=13))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=3)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print("r2스코어", r2)

'''
loss :  16.589523315429688
r2스코어 0.7991998955875229
'''