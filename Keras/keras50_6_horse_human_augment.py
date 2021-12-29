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
    rescale = 1./255,              
    horizontal_flip = True,        
    # vertical_flip= True,           
    width_shift_range = 0.1,       
    height_shift_range= 0.1,       
    # rotation_range= 5,
    zoom_range = 0.3,              
    # shear_range=0.7,
    fill_mode = 'nearest',
    validation_split=0.3          
    )                   # set validation split

xy_train = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/',
    target_size=(150,150),
    batch_size=719,
    class_mode='binary',
    subset='training') # set as training data

xy_test = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/', # same directory as training data
    target_size=(150,150),
    batch_size=308,
    class_mode='binary',
    subset='validation') # set as validation data


# 증폭 데이터 생성
augment_size = 1371
randidx = np.random.randint(719, size = augment_size)

x_augmented = xy_train[0][0][randidx].copy()  #copy() 메모리 생성
y_augmented = xy_train[0][1][randidx].copy()  

x_train = xy_train[0][0].reshape(xy_train[0][0].shape[0],150,150,3)
x_test = xy_test[0][0].reshape(xy_test[0][0].shape[0],150,150,3)

# 증폭한 데이터에 ImageDataGenerator를 사용
augmented_data = train_datagen.flow(x_augmented, y_augmented, batch_size=augment_size, shuffle=False) 
temp_storage = train_datagen.flow(x_augmented, y_augmented, batch_size=30, shuffle=True, save_to_dir="../_temp") 

x = np.concatenate((x_train, augmented_data[0][0]))  # (2000, 150, 150, 3)
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
model.add(Dense(1, activation='sigmoid'))

# #3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])  

start = time.time()
model.fit(x, y, epochs=100, validaiton_split=0.2, steps_per_epoch=100) # (100000/32)
end = time.time() - start
print("걸린 시간 : ", round(end, 2))

#4. 예측 
loss = model.evaluate(xy_test)
print("loss : ", loss)