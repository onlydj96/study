import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler ,RobustScaler, MaxAbsScaler


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# print(datasets.DESCR) # x=(150, 4), y=(150,)
# print(np.unique(y)) # [0, 1, 2]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=4))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3, activation='softmax')) 


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=1000, verbose=1, validation_split=0.2, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss[0])
print("accuracy : ", loss[1])


'''
- 결과값 - 
# 그냥
loss :  0.059321120381355286
accuracy :  0.9666666388511658
# MinMax
loss :  0.07368003576993942
accuracy :  1.0
# Standard
loss :  0.06202506646513939
accuracy :  1.0
# Robuster
loss :  0.061916787177324295
accuracy :  1.0
# MaxAbs
loss :  0.06628243625164032
accuracy :  1.0
'''
