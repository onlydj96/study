import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Dense

tf.compat.v1.set_random_seed(66)

#1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

#2. 모델구성
w1 = tf.get_variable('w1', shape=[3, 3, 1, 64])  # [3, 3, 1]은 kernel_size와 channel 64는 output
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID')



print(w1)  # <tf.Variable 'w1:0' shape=(3, 3, 1, 64) dtype=float32_ref>
print(L1)  # Tensor("Conv2D:0", shape=(?, 26, 26, 64), dtype=float32)