
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.datasets import load_breast_cancer
import time

datasets = load_breast_cancer()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=1004)


# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x에 대한 전처리 완료. y값은 전처리할 필요가 없음

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=30))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

#4. 예측, 결과
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict(x_test)




'''
- 결과값 - 
# 그냥
loss :  [0.27637979388237, 0.8922305703163147]
# MinMax
loss :  [0.10335595905780792, 0.9649122953414917]
# Standard
loss :  [0.08749791234731674, 0.9674185514450073]
# Robuster
loss :  [0.12236346304416656, 0.9598997235298157]
# MaxAbs
loss :  [0.1686561554670334, 0.9624060392379761]

- relu 사용한 결과값 - 
# 그냥
loss :  [0.21426904201507568, 0.9147869944572449]
# MinMax
loss :  [0.427391916513443, 0.9598997235298157]
# Standard
loss :  [0.10484234243631363, 0.9674185514450073]
# Robuster
loss :  [0.13239288330078125, 0.969924807548523]
# MaxAbs
loss :  [0.3143180310726166, 0.9448621273040771]
'''
