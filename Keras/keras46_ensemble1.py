
# 앙상블의 기능과 모델링 구현
# input 2개, output 1개의 앙상블 모델을 만들어라


import numpy as np
from numpy.testing._private.utils import import_nose

#1. 데이터
x1 = np.array([range(100), range(301, 401)])     
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)]) 
x1 = np.transpose(x1)
x2 = np.transpose(x2)

y = np.array(range(1001, 1101)) 

print(x1.shape, x2.shape, y.shape)  # (100, 2) (100, 3) (100,)



from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, random_state=66)

print(x1_train.shape, x1_test.shape)  # (70, 2) (30, 2)
print(x2_train.shape, x2_test.shape)  # (70, 3) (30, 3)
print(y_train.shape, y_test.shape)    # (70,) (30,)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
'''
앙상블로 모델링을 할 때에는 함수형 모델로 구현시켜야한다.
'''

#2-1 모델 1
input1 = Input(shape=(2,))
dense1 = Dense(5, name='dense1', activation='relu')(input1)
dense2 = Dense(7, name='dense2', activation='relu')(dense1)
dense3 = Dense(7, name='dense3', activation='relu')(dense2)
output1 = Dense(7, name='output1', activation='relu')(dense3)

#2-2 모델 2
input2 = Input(shape=(3,))
dense11 = Dense(10, name='dense11', activation='relu')(input2)
dense12 = Dense(10, name='dense12', activation='relu')(dense11)
dense13 = Dense(10, name='dense13', activation='relu')(dense12)
dense14 = Dense(10, name='dense14', activation='relu')(dense13)
output2 = Dense(10, name='output2', activation='relu')(dense14)

#2-3 앙상블
from tensorflow.keras.layers import Concatenate, concatenate   # 두개의 모델을 하나로 합쳐주는 함수
merge1 = concatenate([output1, output2])      # 두개이상일 때 리스트
merge2 = Dense(10, activation='relu')(merge1)
merge3 = Dense(7)(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

#3. 컴파일
model.compile(loss='mse', optimizer='adam')

hist = model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=1)

#4. 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print("loss : ", loss)
results = model.predict([x1_test, x2_test])

from sklearn.metrics import r2_score  
r2 = r2_score(y_test, results)
print("r2스코어", r2)

# import matplotlib.pyplot as plt
# loss = hist.history["loss"]
# mae = hist.history["mse"]
# epochs = range(1, len(loss)+1)

# plt.plot(epochs, loss, 'r--')
# plt.plot(epochs, mae, 'b--')
# plt.grid()
# plt.legend()
# plt.show()

# results = model.predict([x1_test, x2_test])
# print(results)

'''
loss :  3.667269706726074
r2스코어 0.9958058418027128
'''