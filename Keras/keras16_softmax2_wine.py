import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.metrics import accuracy

#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)
# print(np.unique(y))

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

'''
다중 분류는 OneHotEncoding을 해야하기 때문에 이런 방법을 제시한다.
1. categorical : 0컬럼부터 마지막 컬럼까지 빈부분까지 다 포함하여 배열해줌
from tensorflow.keras.utils import to_categorical

2. onehotencoding : 1칼럼부터 정해져있는 컬럼의 개수만큼 배열
from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1,1)
ohe = OneHotEncoder()
y_ohe = ohe.fit_transform(y)
y = y_ohe.toarray()

3. get_dummies : 상동, 행까지 표시
import pandas as pd
y = pd.get_dummies(y) 
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

#2 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3, activation='softmax'))

#3 컴파일
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss[0])
print("accuracy : ", loss[1])

y_pred = model.predict(x_test)
print("예측값 : ", y_pred)

'''
loss :  0.18074281513690948
accuracy :  0.9166666865348816
'''