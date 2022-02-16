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

# Layer 1
w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 32]) 
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
print(L1_maxpool)  # Tensor("MaxPool2d:0", shape=(?, 14, 14, 32), dtype=float32)

# Flatten 
L_flat = tf.reshape(L1_maxpool, [-1, 14*14*32])

# Layer 5
w2 = tf.compat.v1.Variable(tf.random.normal([14*14*32, 32], mean=0, stddev=tf.math.sqrt(2/(14*14*64+32)), name='w5'))
b = tf.compat.v1.Variable(tf.zeros([32]), name='bias')
L2 = tf.nn.relu(tf.compat.v1.matmul(L_flat, w2) + b)

w3 = tf.compat.v1.Variable(tf.random.normal([32, 10], mean=0, stddev=tf.math.sqrt(2/(64+10)), name='w6'))
b2 = tf.compat.v1.Variable(tf.zeros([10], name='bias1'))

output = tf.nn.softmax(tf.matmul(L2, w3) + b2)


# 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(output), axis=1)) # categorical_crossentropy

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(201):
        _, w_val, = sess.run([optimizer, loss], feed_dict = {x:x_train,y:y_train})
        if step % 10 ==0:
            print(step, w_val)

    y_acc_test = sess.run(tf.argmax(y_test, 1))
    predict = sess.run(tf.argmax(sess.run(output, feed_dict={x:x_test}), 1))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y_acc_test),dtype=tf.float32))
    a = sess.run(accuracy,feed_dict = {x:x_test,y:y_test})
    print("\nacc : ", a)
    
# acc :  0.9547