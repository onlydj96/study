import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1. 데이터 분석
path = "./_data/bike/"
train = pd.read_csv(path + "train.csv") # (10886, 12)
test_file = pd.read_csv(path + "test.csv") # (6493, 9)
submit_file = pd.read_csv(path + "sampleSubmission.csv") # (6493, 2)

x = train.drop(columns=['datetime', 'casual', 'registered', 'count'], axis=1) 
y = train['count']
y = np.log1p(y) 
test_file = test_file.drop(columns=['datetime'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))


#3. 컴파일
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor="val_loss", patience=20, mode='min', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=2, validation_split=0.2, callbacks=[es])

#4. 결과
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_pred)
print("r2스코어", r2)

def RMSE(y_test, y_pred): 
    return np.sqrt(mean_squared_error(y_test, y_pred))   
rmse = RMSE(y_test, y_pred) 
print("RMSE : ", rmse)

'''
- 결과값 - 
# 그냥
loss :  1.6055853366851807
r2스코어 0.22331225173064462
RMSE :  1.2671170179833002

# MinMax
loss :  1.543495535850525
r2스코어 0.253347679079608
RMSE :  1.2423750073854294

# Standard
loss :  1.542100429534912
r2스코어 0.2540226921090448
RMSE :  1.2418132941765743

# Robuster
loss :  1.5606135129928589
r2스코어 0.245067020645935
RMSE :  1.2492452174522362

# MaxAbs
loss :  1.543096899986267
r2스코어 0.25354056169461026
RMSE :  1.242214525718404

- relu 사용한 결과값 - 
# 그냥
loss :  1.448490023612976
r2스코어 0.29930591339505386
RMSE :  1.2035322236317154

# MinMax
loss :  1.406860113143921
r2스코어 0.3194438803406129
RMSE :  1.186111367883206

# Standard
loss :  1.3950587511062622
r2스코어 0.3251527373407983
RMSE :  1.181126032919419

# Robuster
loss :  1.404399037361145
r2스코어 0.3206345358343754
RMSE :  1.1850733430891762

# MaxAbs
loss :  1.4300025701522827
r2스코어 0.3082490300861004
RMSE :  1.1958270824328323
'''

###################################### 제출용 제작 ################################################# 


# scaler.fit(test_file)
# test_file = scaler.transform(test_file)
# test_file = scaler.transform(test_file)
# results = model.predict(test_file)
# submit_file['count'] = results


# submit_file.to_csv(path + "bikecount.csv", index=False) # index=False : 인덱스를 행에 추가하지 않는 함수

'''
loss :  1.5408639907836914
r2스코어 0.2546206902456555
RMSE :  1.2413154567015845
'''