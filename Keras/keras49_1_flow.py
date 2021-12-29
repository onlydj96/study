import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
                rescale=1./255, 
                horizontal_flip=True, 
                vertical_flip=True, 
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=5,
                zoom_range=1.2,
                shear_range=0.7,
                fill_mode='nearest'
)


print(x_train[0].shape)                 # (28, 28)
print(x_train[0].reshape(28*28).shape)  # (784,)

augment_size = 50            # 증폭할 데이터 사이즈를 지정

print(np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1).shape)  # (100, 28, 28, 1)

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),
    np.zeros(augment_size),
    batch_size=augment_size,
    shuffle=False
).next()               # batch_size만큼의 한 반복

'''
np.tile : 반복하는 함수로서, 괄호의 첫 번째는 반복할 객체, 두 번째에는 횟수가 나온다. 여기서 횟수는 augument로 지정
np.zeros : y값에 해당
'''

print(type(x_data))  # <class 'tuple'>
print(x_data[0].shape, x_data[1].shape)  # (100, 28, 28, 1) (100,)

import matplotlib.pyplot as plt
plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')
plt.show()