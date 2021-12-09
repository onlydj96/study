import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_diabetes
from tensorflow.keras.utils import to_categorical

#1. 데이터 정제
datasets = load_diabetes()
x = datasets.data # (442, 10) 
y = datasets.target # (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train).reshape(353, 5, 2, 1)
x_test = scaler.fit_transform(x_test).reshape(89, 5, 2, 1)



#2. 모델 구성

model = Sequential()
model.add(Conv2D(10, kernel_size=(2, 2), padding='same', input_shape=(5, 2, 1)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True)
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es])


#4. 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)


from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_pred)
print("r2스코어", r2)

'''
loss :  3193.7666015625
r2스코어 0.4006785347909446
'''