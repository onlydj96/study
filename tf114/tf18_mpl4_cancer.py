
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=66)
x_train = np.array(x_train, dtype='float32')
x_test = np.array(x_test, dtype='float32')
y_test = np.array(y_test, dtype='float32')

x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random.uniform([x_data.shape[1], 64], name='weight'))
b = tf.compat.v1.Variable(tf.zeros([64]), name='bias')  

#2. 모델구성
hidden_layer1 = tf.sigmoid(tf.matmul(x, w) + b)

w1 = tf.compat.v1.Variable(tf.random.uniform([64, 32], name='weight1'))
b1 = tf.compat.v1.Variable(tf.zeros([32], name='bias1'))

hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.uniform([32, 16], name='weight2'))
b2 = tf.compat.v1.Variable(tf.zeros([16], name='bias2'))

hidden_layer3 = tf.nn.relu(tf.matmul(hidden_layer2, w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.uniform([16, 4], name='weight3'))
b3 = tf.compat.v1.Variable(tf.zeros([4], name='bias3'))

hidden_layer4 = tf.nn.relu(tf.matmul(hidden_layer3, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.uniform([4, 1], name='weight3'))
b4 = tf.compat.v1.Variable(tf.zeros([1], name='bias3'))

output = tf.sigmoid(tf.matmul(hidden_layer4, w4) + b4)

loss = -tf.reduce_mean(y*tf.math.log(output) + (1-y)*tf.math.log(1-output))  # binary_crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(1e-9)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(10001):
    loss_val, hy_val, _= sess.run([loss, output, train], feed_dict={x:x_train, y:y_train})

    if epoch % 200 == 0:
        print(epoch, '\t', loss_val)

#4. 평가, 예측
y_predict = tf.cast(output >= 0.5, dtype=tf.float32)

# accuracy score
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_test, y_predict), dtype=tf.float32))
pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_test, y:y_test})

print("Accuracy - \n" , acc)

sess.close()