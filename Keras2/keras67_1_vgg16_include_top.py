
# VGG16의 파라미터 include_top에 대해서

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

# model = VGG16(weights='imagenet', include_top=True,
#               input_shape=(224, 224, 3))

# model.summary()

# print(len(model.weights))
# print(len(model.trainable_weights))

'''
전이학습을 할 경우 input과 output만 수정할 수 있다.
include_top=False를 두면 input과 fc(output)를 수정할 수 있다.
'''

model = VGG16(weights='imagenet', include_top=False, 
              input_shape=(32, 32, 3))

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))
