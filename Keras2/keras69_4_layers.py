from telnetlib import SE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAvgPool2D
from tensorflow.keras.applications import VGG19


vgg19 = VGG19(weights='iamgenet', include_top=False)

model = Sequential()
model.add(vgg19)
model.add(GlobalAvgPool2D())
model.add(Dense(3))

model.summary()

model.layers[0].trainable = False
model.layers[1].trainable = False
model.layers[2].trainable = False