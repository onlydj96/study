# 과제
# 본인 사진으로 predict 하시오
# D:/_data 안에 넣고

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAvgPool2D
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import time

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=0.6,
    zoom_range=0.4,
    # shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.2
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# D:\_data\image\brain

xy_train = train_datagen.flow_from_directory(
    'D:/_data/kaggle/men women/train/',
    target_size=(50, 50),                         # size는 원하는 사이즈로 조정해 줌. 단, 너무 크기 차이가 나면 안좋을 수 있음
    batch_size=10,
    class_mode='binary',
    subset='training',
    shuffle=True,
)       # Found 2317 images belonging to 2 classes.

xy_val = train_datagen.flow_from_directory(
    'D:/_data/kaggle/men women/train/',
    target_size=(50, 50),
    batch_size=10,
    class_mode='binary',
    subset='validation'
)

xy_test = test_datagen.flow_from_directory(
    'D:/_data/kaggle/men women/test/',
    target_size=(50,50),
    batch_size=10,
    class_mode='binary',   
)       # Found 992 images belonging to 2 classes.


#2. 모델구성
pre_train = ResNet101V2(weights='imagenet', include_top=False,
              input_shape=(50, 50, 3))
pre_train.trainable = True

model = Sequential()
model.add(pre_train)
model.add(GlobalAvgPool2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 
learning_rate = 1e-4
optimizer = Adam(learning_rate=learning_rate)

model.compile(loss = "binary_crossentropy", optimizer=optimizer, metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min', verbose=1, factor=0.5)

start = time.time()
hist = model.fit(xy_train, epochs=100, validation_data=xy_val, validation_steps=10, callbacks=[es, reduce_lr]) 
end = time.time() - start

#4. 예측
loss = model.evaluate(xy_test)
print("걸린 시간 : ", round(end, 2))
print("loss, acc : ", loss)

'''
걸린 시간 :  686.9
loss, acc :  [0.5749279856681824, 0.7555282711982727]
'''