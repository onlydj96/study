2
# 넘파일 배열로 저장해놓은 값을 로드한다.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

# np.save("./_save_npy/keras48_4_datasets_x.npy", arr = datasets[0][0])
# np.save("./_save_npy/keras48_4_datasets_y.npy", arr = datasets[0][1])

#1. 데이터
x = np.load("./_save_npy/keras48_4_datasets_x.npy")
y = np.load("./_save_npy/keras48_4_datasets_y.npy")

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


# 스케일러 적용
x_train = x_train.reshape(2647, -1)
x_test = x_test.reshape(662, -1)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(2647, 150, 150, 3)
x_test = x_test.reshape(662, 150, 150, 3)


# 원핫인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(150, 150, 3)))
model.add(MaxPool2D())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es]) 

#4. 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss[-1])

results = model.predict(x_test)  
# print("results : ", results)


# 그래프 시각화

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

import matplotlib.pyplot as plt
# plt.imshow(x_train[0])
# plt.show()

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'r--', label="loss")
plt.plot(epochs, val_loss, 'r:', label="val_loss")
plt.plot(epochs, acc, 'b--', label="acc")
plt.plot(epochs, val_acc, 'b:', label="val_acc")

plt.grid()
plt.legend()
plt.show()