import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale = 1./255,                             
    width_shift_range = 0.1,       
    height_shift_range= 0.1,       
    rotation_range= 0.2,
    zoom_range = 0.1,              
    fill_mode = 'nearest',
    validation_split=0.3          
    )                 

xy_train = train_datagen.flow_from_directory(
    '../_data/image/men_women/',
    target_size=(100,100),
    batch_size=2317,
    class_mode='binary',
    subset='training') 

xy_test = train_datagen.flow_from_directory(
    '../_data/image/men_women/', 
    target_size=(100,100),
    batch_size=992,
    class_mode='binary',
    subset='validation') 


# 증폭 데이터 생성
augment_size = 2683
randidx = np.random.randint(2317, size = augment_size)

x_augmented = xy_train[0][0][randidx].copy()  #copy() 메모리 생성
y_augmented = xy_train[0][1][randidx].copy()  

x_train = xy_train[0][0].reshape(xy_train[0][0].shape[0],100,100,3)
x_test = xy_test[0][0].reshape(xy_test[0][0].shape[0],100,100,3)

# 증폭한 데이터에 ImageDataGenerator를 사용
augmented_data = train_datagen.flow(x_augmented, y_augmented, batch_size=augment_size, shuffle=False) 

x = np.concatenate((x_train, augmented_data[0][0]))  # (5000, 100, 100, 3)
y = np.concatenate((xy_train[0][1], augmented_data[0][1]))  # (5000,)

#2. 모델
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(100, 100, 3)))
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
es = EarlyStopping(monitor='val_loss', patience=15, mode='auto', restore_best_weights=True)

start = time.time()
model.fit(x, y, epochs=100, validation_split=0.2, steps_per_epoch=500, callbacks=[es]) 
end = time.time() - start
print("걸린 시간 : ", round(end, 2))

#4. 예측 
loss = model.evaluate(xy_test)
print("loss : ", loss)

'''
loss :  [0.6821467876434326, 0.5745967626571655]
'''