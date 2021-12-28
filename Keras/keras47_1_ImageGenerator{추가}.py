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
                brightness_range=0.7,
                fill_mode='nearest')


test_datagen = ImageDataGenerator(rescale=1./255)


xy_train = train_datagen.flow_from_directory(
    '../_data/Image/brain/train',
    target_size = (150, 150),
    batch_size = 5,
    class_mode = 'binary',
    shuffle = True,
    save_to_dir='../_temp')

xy_test = test_datagen.flow_from_directory(
    '../_data/Image/brain/test',
    target_size= (150, 150),
    batch_size = 5,
    class_mode = 'binary')

# print(xy_train[32])
print(xy_train[0][0].shape, xy_train[0][1].shape)  # 첫 번째 [0]은 배치 두번째 [0]은 x, [1]은 y

print(type(xy_train)) # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'> 



'''
ab : 80
normal : 80
total : 160 

batch_size : 5

split = 32
'''