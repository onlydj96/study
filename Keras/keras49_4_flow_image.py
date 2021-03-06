import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
                rescale=1./255, 
                horizontal_flip=True, 
                # vertical_flip=True, 
                width_shift_range=0.1,
                height_shift_range=0.1,
                # rotation_range=5,
                zoom_range=0.1,
                # shear_range=0.7,
                fill_mode='nearest'
)


print(x_train[0].shape)                 # (28, 28)
print(x_train[0].reshape(28*28).shape)  # (784,)

augment_size = 40000            # 증폭할 데이터 사이즈를 지정
randidx = np.random.randint(x_train.shape[0], size=augment_size)  
print(x_train.shape[0])     # 60000
print(randidx)              # [53001 50466 35296 ... 49632 49187  4642]
print(np.min(randidx), np.max(randidx))  # 0 59996

# 증폭 생성
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)      # (40000, 28, 28)
print(y_augmented.shape)      # (40000,)

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size=augment_size,
                                 shuffle=False
).next()[0]

# 증폭된 데이터를 합치다.
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)          # (100000, 28, 28) (100000,)


# 1. x_augmented 10개와 x_train 10개를 비교하는 이미지 출력할 것 
# subplot(2, 10, ?) 사용

# print(x_augmented[randidx])

import matplotlib.pyplot as plt
plt.figure(figsize=(7, 7))
for i in range(19):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_augmented[i], 'gray')
    
rows = 10
cols = 2

plt.show()