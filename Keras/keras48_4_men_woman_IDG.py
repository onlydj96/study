# 과제
# 본인 사진으로 predict 하시오
# D:/_data 안에 넣고

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
                rescale=1./255, 
                horizontal_flip=True, 
                vertical_flip=True, 
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=5,
                zoom_range=1.2,
                shear_range=0.7,
                fill_mode='nearest')


datasets = datagen.flow_from_directory(
    'D:\_data\Image\men_women',
    target_size = (150, 150),
    batch_size = 4000,
    class_mode = 'binary',
)

print(type(datasets)) # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(datasets[0][0].shape)

np.save("./_save_npy/keras48_4_datasets_x.npy", arr = datasets[0][0])
np.save("./_save_npy/keras48_4_datasets_y.npy", arr = datasets[0][1])