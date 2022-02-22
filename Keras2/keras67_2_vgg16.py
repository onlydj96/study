from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

vgg16 = VGG16(weights='imagenet', include_top=False,
              input_shape=(32, 32, 3))
vgg16.trainable = False  # vgg16의 레이어에 대해서는 훈련을 안시킨다.(가중치 동결)

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()
print(len(model.trainable_weights))

