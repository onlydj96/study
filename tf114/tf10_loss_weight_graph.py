
# loss와 weight에 대한 그래프 시각화

import tensorflow as tf, os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(77)

x = [1, 2, 3]
y = [1, 2, 3]
w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis-y))

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 33):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w:curr_w})
        w_history.append(curr_w)
        loss_history.append(curr_loss)


plt.plot(w_history, loss_history)
plt.xlabel('Weight')
plt.ylabel('Loss')
plt.show()