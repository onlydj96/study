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

'''
rescale : scale 해주는 것
horizontal_flip : 가로로 반을 접은 것
vertical_flip : 세로로 반을 접은 것 (위랑 반대일 수도?)
width_shift_range : 가로로 이동
height_shift_range : 세로로 이동(위로)
zoom_range : 확대
rotation_range :
shear_range :
brightness_range= : 밝기
fill_mode : 이동해서 공백이 된 부분을 어떻게 채우는 지
'''    


test_datagen = ImageDataGenerator(rescale=1./255)

'''
훈련데이터는 데이터 증폭을 할 수 있으나 테스트 데이터는 데이터 증폭을 하지 않는다.
'''

# D:/_data/Image/brain

xy_train = train_datagen.flow_from_directory(
    '../_data/Image/brain/train',
    target_size = (150, 150),
    batch_size = 5,
    class_mode = 'binary',
    shuffle = True)

xy_test = test_datagen.flow_from_directory(
    '../_data/Image/brain/test',
    target_size= (150, 150),
    batch_size = 5,
    class_mode = 'binary')

# print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000011C6D084F70>

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