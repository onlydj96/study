
# 모델링 구성

import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
                rescale=1./255, 
                horizontal_flip=True, 
                # vertical_flip=True, 
                width_shift_range=0.1,
                height_shift_range=0.1,
                # rotation_range=5,
                zoom_range=0.1,
                # shear_range=0.7,
                fill_mode='nearest'
)

print(x_train[0].shape)                 # (28, 28)
print(x_train[0].reshape(28*28).shape)  # (784,)

augment_size = 40000            # 증폭할 데이터 사이즈를 지정
randidx = np.random.randint(x_train.shape[0], size=augment_size)  
print(x_train.shape[0])     # 60000
print(randidx)              # [53001 50466 35296 ... 49632 49187  4642]
print(np.min(randidx), np.max(randidx))  # 0 59996


# 증폭 생성
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)      # (40000, 28, 28)
print(y_augmented.shape)      # (40000,)

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False
).next()[0]
'''
10000
5batch => 2000
1epoch = 2000
batch_size = 200 
1epoch = 10

'''
# 증폭된 데이터를 합치다.
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)          # (100000, 28, 28) (100000,)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(3,3), strides=2, padding='same', input_shape=(28, 28, 1)))   # 27, 27, 6  
model.add(MaxPool2D())
model.add(Conv2D(5, (3,3), activation='relu'))   # 7, 7, 5
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True)
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])


#4. 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

print("r2스코어", r2)
