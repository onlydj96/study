

# http://www.kaggle.com/c/dogs-vs-cats/data

from typing import Sequence
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping

train_datagen = ImageDataGenerator(
                rescale=1./255, 
                horizontal_flip=True, 
                vertical_flip=True, 
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=5,
                zoom_range=1.2,
                shear_range=0.7,
                fill_mode='nearest')


test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    'D:/_data/Image/cat_dog/training_set',
    target_size = (150, 150),
    batch_size = 50,
    class_mode = 'categorical',
    shuffle = True)

xy_test = test_datagen.flow_from_directory(
    'D:/_data/Image/cat_dog/test_set',
    target_size= (150, 150),
    batch_size = 50,
    class_mode = 'categorical')

print(type(xy_train)) # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(xy_train[0][0].shape)


#2. 모델
model = Sequential()

model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
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

#3. 컴파일 
model.compile(loss = "categorical_crossentropy", optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
hist = model.fit_generator(xy_train, epochs=100, steps_per_epoch=100, validation_data=xy_train, validation_steps=40, callbacks=[es]) 

#4. 예측
pic_path = '../_data/Image/test_file/dog.jpg'
img = image.load_img(pic_path, target_size=(150,150))
  
plt.imshow(img, 'gray')
plt.show()

img = image.load_img(pic_path, target_size=(150,150))
img = img_to_array(img)
img = img.reshape((1,) + img.shape)
print(img.shape)

loss = model.evaluate_generator(xy_test)
print("loss : ", loss[-1])
results = model.predict(img)
print("results : ", results)


'''
loss :  [0.5667693614959717, 0.7172515988349915]
'''

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