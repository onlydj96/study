import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical

#1. 데이터 정제
datasets = load_iris()
x = datasets.data # (150. 4)
y = datasets.target # (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

# print(x_train.shape, x_test.shape)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train).reshape(120, 2, 2, 1)
x_test = scaler.fit_transform(x_test).reshape(30, 2, 2, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(1, 1), padding='same', input_shape=(2, 2, 1)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(3, activation='softmax'))

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es])


#4. 예측
loss = model.evaluate(x_test, y_test)
print('loss, accuracy : ', loss)

# acc = str(round(loss[1], 4))
# model.save("./_save/cnn_iris_{}.h5".format(acc))

'''
loss, accuracy :  [0.25957953929901123, 0.9333333373069763]
'''
