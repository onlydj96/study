from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
import time

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))
vgg16.trainable = False  # vgg16의 레이어에 대해서는 훈련을 안시킨다.(가중치 동결)

model = Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=15)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[es, reduce_lr], validation_split=0.2)
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print("걸린 시간 : ", round(end, 2))
print('loss, acc ', loss)


'''
1. False, Flatten -
걸린 시간 :  287.48
loss, acc  [1.1683835983276367, 0.6080999970436096]

2. True, Flatten -
걸린 시간 :  467.77
loss, acc  [1.094401478767395, 0.7903000116348267]

3. False, GAP - 
걸린 시간 :  284.36
loss, acc  [1.1427831649780273, 0.6128000020980835]

4. True, GAP -
걸린 시간 :  519.85
loss, acc  [1.1903995275497437, 0.7562000155448914]
'''