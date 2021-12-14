
# Dropout 함수의 의미 및 구현
    
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout   # layers 에 Dropout 모델을 추가
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston

datasets = load_boston()
x = datasets.data 
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=1004)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dropout(0.5))   # Dense(100)에 50% 적용됨
model.add(Dense(50, activation='relu')) 
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))    # Dense(10)에 20% 적용됨
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

'''
Dropout이란 nod에 일부를 없애줌으로써 연산력을 높임과 동시에 정확도를 상승시킬 수 있다.(상황에 따라 정확도는 떨어질 수 있다)
Dropout 모델링은 바로 위에 Dense의 nod에 적용된다
'''

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") 

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'      
model_path = "".join([filepath, '1_boston_', datetime, '_', filename])

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', verbose=1, mode='min', save_best_only=True, filepath=model_path)
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

#4. 예측, 결과
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_pred)
print("r2스코어", r2)

'''
loss :  23.617778778076172
r2스코어 0.7391462585360851
'''