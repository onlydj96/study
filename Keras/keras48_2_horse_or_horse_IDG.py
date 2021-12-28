

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                rescale=1./255, 
                horizontal_flip=True, 
                vertical_flip=True, 
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=5,
                zoom_range=1.2,
                shear_range=0.7,
                brightness_range=0.7,
                fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/Image/horse-or-human',
    target_size = (150, 150),
    batch_size = 5,
    class_mode = 'binary',
    shuffle = True)

xy_test = test_datagen.flow_from_directory(
    '../_data/Image/horse-or-human',
    target_size= (150, 150),
    batch_size = 5,
    class_mode = 'binary')

print(type(xy_train)) # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(xy_train[0][0].shape)

np.save("./_save_npy/keras48_2_datasets_x.npy", arr = datasets[0][0])
np.save("./_save_npy/keras48_2_datasets_y.npy", arr = datasets[0][1])

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
loss = model.evaluate(x_test, y_test)
print("loss : ", loss[-1])
results = model.predict(x_test)
print("results : ", results[-1])


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