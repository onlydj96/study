import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x_predict = [[1, 11, 7, 9]]

#2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.compat.v1.Variable(tf.random.normal([4, 3]), name='weight')
b = tf.compat.v1.Variable(tf.random.normal([1, 3]), name='bias')  

#2. 모델
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))    # categorical_crossentropy
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

#3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for epoch in range(20001):
        _, loss_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})
        
        if epoch % 200 == 0:
            print(epoch, '\t', loss_val)

#4. 평가, 예측
    results = sess.run(hypothesis, feed_dict={x:x_data})
    print(sess.run(tf.math.argmax(results, 1)))


y_data = tf.math.argmax(y_data, 1)
print(y_data)

# accuracy score
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_data, results), dtype=tf.float32))
print(accuracy)