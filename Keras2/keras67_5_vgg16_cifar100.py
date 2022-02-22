from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
import time

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

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
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='amin', verbose=1, factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, callbacks=[es, reduce_lr], validation_split=0.2)
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print("걸린 시간 : ", round(end, 2))
print('loss, acc ', loss)


'''
1. False, Flatten -
걸린 시간 :  167.41
loss, acc  [3.7538840770721436, 0.35899999737739563]

2. True, Flatten -
걸린 시간 :  316.53
loss, acc  [4.605484962463379, 0.009999999776482582]

3. False, GAP - 
걸린 시간 :  167.22
loss, acc  [3.8341753482818604, 0.3573000133037567]

4. True, GAP
걸린 시간 :  332.13
loss, acc  [4.705484962463441, 0.009999989776482573]
'''