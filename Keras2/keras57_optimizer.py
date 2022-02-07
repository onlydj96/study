
# optimizer의 모델과 learning_rate 조절

import numpy as np

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 3, 5, 4, 7, 6, 7, 11, 9, 7])

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax, RMSprop, SGD, Nadam

learning_rate = 0.00001

# optimizer = Adam(learning_rate=learning_rate)      # loss :  2.52 lr :  1e-05 결과물 :  [[11.136969]]
# optimizer = Adadelta(learning_rate=learning_rate)  # loss :  2.52 lr :  0.01 결과물 :  [[11.240752]]
# optimizer = Adagrad(learning_rate=learning_rate)   # loss :  2.33 lr :  0.001 결과물 :  [[11.063637]]
# optimizer = Adamax(learning_rate=learning_rate)    # loss :  2.35 lr :  0.0003 결과물 :  [[11.399364]]
# optimizer = RMSprop(learning_rate=learning_rate)   # loss :  2.53 lr :  1e-05 결과물 :  [[11.155262]]
# optimizer = SGD(learning_rate=learning_rate)       # loss :  2.58 lr :  1e-05 결과물 :  [[11.269703]]
optimizer = Nadam(learning_rate=learning_rate)     # loss :  2.53 lr :  1e-05 결과물 :  [[11.204746]]
 
model.compile(loss='mse', optimizer=optimizer)
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
pred = model.predict([11])

print("loss : ", round(loss, 2), "lr : ", learning_rate, "결과물 : ", pred)