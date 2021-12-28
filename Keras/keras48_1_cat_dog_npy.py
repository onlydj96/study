
# 넘파일 배열로 저장해놓은 값을 로드한다.

import numpy as np
from tensorflow.keras.utils import to_categorical
# np.save("./_save_npy/keras48_1_train_x.npy", arr = xy_train[0][0])
# np.save("./_save_npy/keras48_1_train_y.npy", arr = xy_train[0][1])
# np.save("./_save_npy/keras48_1_test_x.npy", arr = xy_test[0][0])
# np.save("./_save_npy/keras48_1_test_y.npy", arr = xy_test[0][1])

#1. 데이터
x_train = np.load("./_save_npy/keras48_1_train_x.npy")
y_train = np.load("./_save_npy/keras48_1_train_y.npy")

x_test = np.load("./_save_npy/keras48_1_test_x.npy")
y_test = np.load("./_save_npy/keras48_1_test_y.npy")

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
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.2, callbacks=[es]) 

#4. 예측
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
pic_path = '../_data/Image/test_file/dog.jpg'
img = image.load_img(pic_path, target_size=(150,150))
  
plt.imshow(img, 'gray')
plt.show()

img = image.load_img(pic_path, target_size=(150,150))
img = img_to_array(img)
img = img.reshape((1,) + img.shape)
print(img.shape)

loss = model.evaluate(x_test, y_test)
print("loss : ", loss[-1])
results = model.predict(img)
print("results : ", results)



# 그래프 시각화
import matplotlib.pyplot as plt

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

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