import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
train_datagen = ImageDataGenerator(
                rescale=1./255, 
                horizontal_flip=True, 
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=0.2,
                zoom_range=1.2,
                fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/Image/cat_dog/training_set',
    target_size = (150, 150),
    batch_size = 4000,
    class_mode = 'binary',
    shuffle = True)

xy_test = test_datagen.flow_from_directory(
    '../_data/Image/cat_dog/test_set',
    target_size= (150, 150),
    batch_size = 300,
    class_mode = 'binary')


# 증폭 데이터 생성
augment_size = 5000
randidx = np.random.randint(160, size = augment_size)

x_augmented = xy_train[0][0][randidx].copy()  #copy() 메모리 생성
y_augmented = xy_train[0][1][randidx].copy()  

x_train = xy_train[0][0].reshape(xy_train[0][0].shape[0],150,150,3)
x_test = xy_test[0][0].reshape(xy_test[0][0].shape[0],150,150,3)


# 증폭한 데이터에 ImageDataGenerator를 사용
augmented_data = train_datagen.flow(x_augmented, y_augmented, batch_size=augment_size, shuffle=False) 
temp_storage = train_datagen.flow(x_augmented, y_augmented, batch_size=30, shuffle=True, save_to_dir="../_temp") 

x = np.concatenate((x_train, augmented_data[0][0]))  # (500, 150, 150, 3)
y = np.concatenate((xy_train[0][1], augmented_data[0][1]))  # (500,)

#2. 모델
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
model.add(Dense(10, activation='softmax'))

# #3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])  

start = time.time()
model.fit(x, y, epochs=100, steps_per_epoch=100) # (100000/32)
end = time.time() - start
print("걸린 시간 : ", round(end, 2))

#4. 예측 
loss = model.evaluate(xy_test)
print("loss : ", loss)