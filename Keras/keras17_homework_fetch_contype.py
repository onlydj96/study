import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn import datasets
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.metrics import accuracy
import time

#1. 데이터
datasets = fetch_covtype()
x = datasets.data # (581012, 54)
y = datasets.target # (581012, )

# print(x.shape, y.shape)
# print(datasets.DESCR)
# print(datasets.feature_names)
# print(np.unique(y)) # 7


import pandas as pd
y = pd.get_dummies(y)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=54))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(7, activation='softmax'))

#3. 컴파일
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es])
end = time.time() - start
print("걸린 시간 : ", round(end, 2), "초")

# #4. 예측, 결과
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss[0])
# print('accuracy : ', loss[1])

'''
loss :  0.647487461566925
accuracy :  0.720738708972930
'''