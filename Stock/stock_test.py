import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

ss_stock = pd.read_csv("D:/_data/stock predict/삼성전자.csv", thousands=',', encoding='CP949')

# 삼성증권 데이터 전처리

x = ss_stock.drop(columns=['일자', '종가', '전일비', '거래량', 'Unnamed: 6', '등락률', '금액(백만)', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1)
y = ss_stock['종가']

x = x.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.7, random_state=1)

scaler = MinMaxScaler()
x = scaler.fit_transform(x).astype(float)
y = y/100000

x_train = x_train.reshape(896, 4, 1)
x_test = x_test.reshape(156, 4, 1)
x_val = x_val.reshape(68, 4, 1)


print(x_train.shape, x_test.shape, x_val.shape)


model = Sequential()
model.add(LSTM(100, input_shape=(4, 1)))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))


