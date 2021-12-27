

# http://www.kaggle.com/c/dogs-vs-cats/data

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                rescale=1./255, 
                horizontal_flip=True, 
                vertical_flip=True, 
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=5,
                zoom_range=1.2,
                shear_range=0.7,
                fill_mode='nearest')


test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    'D:/_data/Image/cat_dog/training_set',
    target_size = (150, 150),
    batch_size = 4000,
    class_mode = 'binary',
    shuffle = True)

xy_test = test_datagen.flow_from_directory(
    'D:/_data/Image/cat_dog/test_set',
    target_size= (150, 150),
    batch_size = 5000,
    class_mode = 'binary')

print(type(xy_train)) # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(xy_train[0][0].shape)

np.save("./_save_npy/keras48_1_train_x.npy", arr = xy_train[0][0])
np.save("./_save_npy/keras48_1_train_y.npy", arr = xy_train[0][1])
np.save("./_save_npy/keras48_1_test_x.npy", arr = xy_test[0][0])
np.save("./_save_npy/keras48_1_test_y.npy", arr = xy_test[0][1])