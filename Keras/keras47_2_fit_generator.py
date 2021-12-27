import numpy as np
from numpy.lib.function_base import percentile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
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
    '../_data/Image/brain/train',
    target_size = (150, 150),
    batch_size = 5,
    class_mode = 'binary',
    shuffle = True)

xy_test = test_datagen.flow_from_directory(
    '../_data/Image/brain/test',
    target_size= (150, 150),
    batch_size = 5,
    class_mode = 'binary')

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
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
hist = model.fit_generator(xy_train, epochs=100, steps_per_epoch=32, validation_data=xy_train, validation_steps=4) 

'''
fit_generator : 분리되지 않은 xy_train을 바로 지정 가능
steps_per_epochs= : 전체데이터/batch를 의미한다.
validation_ : 조사해보자...
'''


# 그래프 시각화
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

import matplotlib.pyplot as plt

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

'''
loss :  0.15039998292922974
val_loss :  0.2737499177455902
acc :  0.96875
val_acc :  0.8500000238418579
'''