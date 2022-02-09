
# y = wx + b, keras 1번 예제와 동일한 문제

import tensorflow as tf
tf.set_random_seed(66)

x_train = [1, 2, 3]
y_train = [1, 2, 3]

w = tf.Variable(1, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)

#2. 모델 구성
hypothesis = x_train * w + b   # y = wx + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))   # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))