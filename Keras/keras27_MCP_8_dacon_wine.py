import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import ModelCheckpoint

#1. 데이터 분석
path = "../_data/kaggle/bike/"
train = pd.read_csv(path + "train.csv") # (10886, 12)
test_file = pd.read_csv(path + "test.csv") # (6493, 9)
submit_file = pd.read_csv(path + "sampleSubmission.csv") # (6493, 2)

x = train.drop(columns=['datetime', 'casual', 'registered', 'count'], axis=1) 
y = train['count']
y = np.log1p(y) 
test_file = test_file.drop(columns=['datetime'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)


scaler = StandardScaler()
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

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") 

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'      
model_path = "".join([filepath, '8_dacon_wine_', datetime, '_', filename])


es = EarlyStopping(monitor="val_loss", patience=20, mode='min', verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True, filepath=model_path)
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=2, validation_split=0.2, callbacks=[es, mcp])

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

