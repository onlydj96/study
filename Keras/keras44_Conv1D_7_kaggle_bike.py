import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPool1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler ,RobustScaler, MaxAbsScaler
import time


#1. 데이터 전처리
path = "../_data/kaggle/bike/"
train = pd.read_csv(path + "train.csv") # (10886, 12)
test_file = pd.read_csv(path + "test.csv") # (6493, 9)
submit_file = pd.read_csv(path + "sampleSubmission.csv") # (6493, 2)

x = train.drop(columns=['datetime', 'casual', 'registered', 'count'], axis=1)
y = train['count']
test_file = test_file.drop(columns=['datetime'], axis=1) 

y = np.log1p(y)  # 데이터 숫자를 작게 만듦

# data split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)
print(x_train.shape, x_test.shape)  # (8708, 8) (2178, 8)

# data scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  # (8108, 8)
x_test = scaler.fit_transform(x_test) # (2178, 8)

# 데이터 RNN 형식으로 바꿈
x_train = x_train.reshape(8708, 8, 1)
x_test = x_test.reshape(2178, 8, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(100, 2, input_shape=(8, 1)))
model.add(MaxPool1D())
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(10))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. 컴파일
model.compile(loss='mse', optimizer='adam')
start = time.time()

model.fit(x_train, y_train, epochs=50, validation_split=0.2)
end = time.time() - start
print("걸린 시간 : ", round(end, 3))

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
LSTM
걸린 시간 :  35.315
loss :  1.422283411026001
r2스코어 0.311983070728732
RMSE :  1.1925952047282644

Conv1D
걸린 시간 :  11.177
loss :  1.471269965171814
r2스코어 0.2882861618863808
RMSE :  1.2129592241715217
'''
