

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAvgPool2D, Dropout
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import time


train_datagen = ImageDataGenerator(rescale=1./255)

batch_num = 128
xy_train = train_datagen.flow_from_directory(
    'D:/_data/kaggle/rps/train',
    target_size=(100, 100),                         # size는 원하는 사이즈로 조정해 줌. 단, 너무 크기 차이가 나면 안좋을 수 있음
    batch_size=batch_num,
    class_mode='categorical',
    shuffle=True,
) 

xy_test = train_datagen.flow_from_directory(
    'D:/_data/kaggle/rps/test',
    target_size=(100, 100),
    batch_size=batch_num,
    class_mode='categorical',   
)       # Found 756 images belonging to 3 classes.


np.save("D:/_save/rps_x_train.npy", arr = xy_train[0][0])
np.save("D:/_save/rps_y_train.npy", arr = xy_train[0][1])
np.save("D:/_save/rps_x_test.npy", arr = xy_test[0][0])
np.save("D:/_save/rps_y_test.npy", arr = xy_test[0][1])

x_train = np.load("D:/_save/rps_x_train.npy")
y_train = np.load("D:/_save/rps_y_train.npy")

x_test = np.load("D:/_save/rps_x_test.npy")
y_test = np.load("D:/_save/rps_y_test.npy")

#2. 모델 구성
pre_train = VGG19(weights='imagenet', include_top=False,
              input_shape=(100, 100, 3))
pre_train.trainable = True

model = Sequential()
model.add(pre_train)
model.add(GlobalAvgPool2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

#3. 컴파일 
learning_rate = 1e-4
optimizer = Adam(learning_rate=learning_rate)

model.compile(loss = "categorical_crossentropy", optimizer=optimizer, metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min', verbose=1, factor=0.5)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split=0.2, callbacks=[es, reduce_lr]) 
end = time.time() - start

#4. 예측
loss = model.evaluate(x_test, y_test)
print("걸린 시간 : ", round(end, 2))
print("loss, acc : ", loss)

'''
걸린 시간 :  125.68
loss, acc :  [0.5728682279586792, 0.9453125]
'''