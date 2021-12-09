########################################
# 각각의 Scaler의 특성과 정의를 정리하시오
########################################

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, \
                                  StandardScaler, RobustScaler, MaxAbsScaler    # 4가지의 Scaler 함수
from sklearn.datasets import load_boston

datasets = load_boston()
x = datasets.data 
y = datasets.target

'''
데이터 전처리를 간단하게 하는법
# print(np.min(x), np.max(x))  # 0.0 711.0   
# x = x/711
# x = x/np.max(x)

'''

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=1004)

print(x.shape)

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x에 대한 전처리 완료. y값은 전처리할 필요가 없음

'''
scaler는 데이터를 전처리하는데 쓰이는 함수이다. 데이터의 값이 너무 클때 머신에서의 계산의 속도와 정확도가 낮아지기 때문에
데이터의 최솟값과 최대값을 기준으로 0과 1 사이로 모든 값들을 바꾼다. train의 데이터를 기준으로 최소, 최대값을 정하여 scaling하고,
test의 값들은 train에서 바꾼 0과 1기준에 맞추어 scaling한다.  
'''

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(50, activation='relu')) # relu함수는 layers에 y=wx+b라는 함수에 음수가 나올때, 그것을 0으로 바꿔준다.
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
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_pred)
print("r2스코어", r2)


'''
- 결과값 - 
# 그냥
loss :  49.48818588256836
r2스코어 0.45341265883137793
# MinMax
loss :  24.4689998626709
r2스코어 0.7297446981202182
# Standard
loss :  16.455062866210938
r2스코어 0.8182570634097548
# Robuster
loss :  25.892732620239258
r2스코어 0.7140198470710837
# MaxAbs
loss :  31.447059631347656
r2스코어 0.6526733718987099


- relu 사용한 결과값 - 
# 그냥
loss :  34.036842346191406
r2스코어 0.6240697361595857
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
loss :  20.296234130859375
r2스코어 0.7758320830342779
'''
