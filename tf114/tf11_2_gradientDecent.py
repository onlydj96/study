
# r2스코어 구하기

import tensorflow as tf, os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


x_train_data = [1,2,3]
y_train_data = [1,2,3]
x_test_data = [4,5,6]
y_test_data = [4,5,6]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable(tf.random.normal([1]), name = 'weight')

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))
lr = 0.2

# 기울기
gradient = tf.reduce_mean((x * w - y) * x)


descent = w - lr * gradient   
update = w.assign(descent)        # w = w - lr * gradient  # assign 할당함수  ===> gradientdescentoptimizer

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

w_history = []
loss_history = []

for step in range(10):

    _, loss_v, w_v = sess.run([update, loss, w], feed_dict = {x: x_train_data, y:y_train_data})
    print(step, '\t', loss_v, '\t', w_v)
    
from sklearn.metrics import r2_score

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

y_predict = x_test * w_v
pred = sess.run(y_predict, feed_dict={x_test:x_test_data})
sess.close()

print(r2_score(pred, y_test_data))
