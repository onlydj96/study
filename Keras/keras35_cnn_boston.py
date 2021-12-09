import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
from tensorflow.keras.utils import to_categorical

#1. 데이터 정제
datasets = load_boston()
x = datasets.data
y = datasets.target

import pandas as pd
x_refine = pd.DataFrame(x, columns=datasets.feature_names)
x_cnn = x_refine.drop(columns=['CHAS'], axis=1) # (506, 12)
x_cnn = x_cnn.to_numpy()

# print(type(x_cnn))
# print(x_cnn)
# print(x_cnn.corr())


x_train, x_test, y_train, y_test = train_test_split(x_cnn, y, train_size=0.8, random_state=1)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train).reshape(404, 4, 3, 1)
x_test = scaler.fit_transform(x_test).reshape(102, 4, 3, 1)



#2. 모델 구성

model = Sequential()
model.add(Conv2D(10, kernel_size=(2, 2), padding='same', input_shape=(4, 3, 1)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True)
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])


#4. 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)


from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_pred)
print("r2스코어", r2)

'''
loss :  29.36621856689453
r2스코어 0.7028534361664693
'''


# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(10,10))
# sns.heatmap(data=x_refine.corr(), square=True, annot=True, cbar=True)
# plt.show
