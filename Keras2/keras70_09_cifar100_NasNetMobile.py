
# ValueError: When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3).  Received: input_shape=(32, 32, 3)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input
import time

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

NNM = NASNetMobile(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))
NNM.trainable = False 

model = Sequential()
model.add(NNM)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='amin', verbose=1, factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, callbacks=[es, reduce_lr], validation_split=0.2)
end = time.time() - start

loss = model.evaluate(x_test, y_test)
print("걸린 시간 : ", round(end, 2))
print('loss, acc ', loss)


'''

'''