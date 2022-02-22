
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = "D:/_data/winequality/"
datasets = pd.read_csv(path + 'winequality-white.csv', index_col = None, header = 0, sep=';') # 분리

datasets = datasets.values #  pandas --> numpy로 바꿔주기
#print(type(datasets)) # <class 'numpy.ndarray'>

x = datasets[:,:11]  # 모든 행, 10번째까지
y = datasets[:, 11]  # 모든행, 11번째 열이 y 

print(np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8, stratify = y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

print(y_train.shape)

#2. 모델
model = Sequential()
model.add(Dense(128, input_dim=11))
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es, reduce_lr])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

pred = model.predict(x_test)
print(f1_score(x_test, pred))
# loss :  [0.25527939200401306, 0.550000011920929]
