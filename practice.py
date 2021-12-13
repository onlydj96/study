import numpy as np
import pandas as pd
from pandas.core.reshape.reshape import get_dummies
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits

datasets = load_digits()

# print(datasets.DESCR)
# print(datasets.data.shape)

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (1797, 64) (1797,)
print(pd.unique(y)) # [0 1 2 3 4 5 6 7 8 9]

import matplotlib.pyplot as plt
plt.matshow(datasets.images[0])
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train).reshape(1437, 8, 8, 1)
x_test = scaler.fit_transform(x_test).reshape(360, 8, 8, 1)

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)  # (1437, 64) (360, 64) (1437,) (360,)

y_train = get_dummies(y_train)
y_test = get_dummies(y_test)

#2. 모델구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(2, 2), input_shape=(8, 8, 1)))
model.add(MaxPool2D())
model.add(Conv2D(5, kernel_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))


#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=1, validation_split=0.2, callbacks=[es])

#4. 예측
loss = model.evaluate(x_test, y_test)
print('loss, accuracy : ', loss)


