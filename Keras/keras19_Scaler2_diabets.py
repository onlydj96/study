########################################
# 각각의 Scaler의 특성과 정의를 정리하시오
########################################

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data 
y = datasets.target


# print(np.min(x), np.max(x))  
# x = x/711
# x = x/np.max(x)

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
model.add(Dense(100, input_dim=10))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

#4. 예측, 결과
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)

print("r2스코어", r2)



'''
- 결과값 - 
# 그냥
loss :  44.16288757324219
r2스코어 0.5122295659855384
# MinMax
loss :  23.10694122314453
r2스코어 0.7447883615607933
# Standard
loss :  15.5115385055542
r2스코어 0.8286781186312449
# Robuster
loss :  19.066755294799805
r2스코어 0.7894114413798203
# MaxAbs
loss :  26.3043212890625
r2스코어 0.7094739079797971

- relu 사용한 결과값 - 
# 그냥
loss :  29.571008682250977
r2스코어 0.6733940299898684
# MinMax
loss :  26.091548919677734
r2스코어 0.7118239656710365
# Standard
loss :  26.640918731689453
r2스코어 0.7057562500596156
# Robuster
loss :  20.57558822631836
r2스코어 0.7727466639628486
# MaxAbs
loss :  21.019636154174805
r2스코어 0.7678422190500289
'''
