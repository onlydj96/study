import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
x = np.array(range(17))
y = np.array(range(17))

x_train = x[:11] 
y_train = y[:11]
x_test = x[11:14] 
y_test = y[11:14]
x_val = x[14:17]
y_val = y[14:17]

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1, verbose=2, 
          validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)

'''
loss :  1.8189894035458565e-12
17의 예측값 :  [[17.]]
'''