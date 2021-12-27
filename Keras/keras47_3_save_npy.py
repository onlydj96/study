
# 넘파이 배열로 값을 저장하기

import numpy as np
from numpy.lib.function_base import percentile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
train_datagen = ImageDataGenerator(
                rescale=1./255, 
                # horizontal_flip=True, 
                # vertical_flip=True, 
                # width_shift_range=0.1,
                # height_shift_range=0.1,
                # rotation_range=5,
                # zoom_range=1.2,
                # shear_range=0.7,
                # fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../_data/Image/brain/train',
    target_size = (150, 150), 
    batch_size = 200,                   # batch_size가 
    class_mode = 'binary',
    shuffle = True
)

xy_test = test_datagen.flow_from_directory(
    '../_data/Image/brain/test',
    target_size= (150, 150),
    batch_size = 200,
    class_mode = 'binary'
)

print(xy_train[0][0].shape) 
print(xy_test[0][0].shape) 

np.save("./_save_npy/keras47_3_train_x.npy", arr = xy_train[0][0])
np.save("./_save_npy/keras47_3_train_y.npy", arr = xy_train[0][1])
np.save("./_save_npy/keras47_3_test_x.npy", arr = xy_test[0][0])
np.save("./_save_npy/keras47_3_test_y.npy", arr = xy_test[0][1])