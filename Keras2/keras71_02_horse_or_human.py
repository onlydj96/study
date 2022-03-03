import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, GlobalAvgPool2D
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.optimizers import Adam
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale = 1./255,              
    validation_split=0.2          
    ) 

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    'D:/_data/kaggle/horse-or-human/train',
    target_size = (100,100),
    batch_size = 32,
    class_mode = 'binary',
    subset = 'training',
    shuffle = True) 

xy_val = train_datagen.flow_from_directory(
    'D:/_data/kaggle/horse-or-human/train',
    target_size = (100, 100),
    batch_size = 32,
    class_mode = 'binary',
    subset = 'validation')  

xy_test = test_datagen.flow_from_directory(
    'D:/_data/kaggle/horse-or-human/validation', # same directory as training data
    target_size = (100,100),
    batch_size = 32,
    class_mode = 'binary')

#2. 모델
pre_train = ResNet101(weights='imagenet', include_top=False,
              input_shape=(100, 100, 3))
pre_train.trainable = True

model = Sequential()
model.add(pre_train)
model.add(GlobalAvgPool2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 
learning_rate = 1e-4
optimizer = Adam(learning_rate=learning_rate)

model.compile(loss = "binary_crossentropy", optimizer=optimizer, metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='amin', verbose=1, factor=0.5)

start = time.time()
hist = model.fit(xy_train, epochs=100, validation_data=xy_val, validation_steps=5, callbacks=[es, reduce_lr]) 
end = time.time() - start

#4. 예측
loss = model.evaluate(xy_test)
print("걸린 시간 : ", round(end, 2))
print("loss, acc : ", loss)

'''
걸린 시간 :  132.37
loss, acc :  [0.41142579913139343, 0.84765625]
'''