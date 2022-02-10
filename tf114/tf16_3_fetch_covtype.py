import numpy as np
import tensorflow as tf
import os
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

#1. 데이터

datasets = fetch_covtype()
x_data = datasets.data     # (178, 54)
y_data = datasets.target   # (178,)
y_data = datasets.target.reshape(-1,1)

ohe = OneHotEncoder()
ohe.fit(y_data)
y_data = ohe.transform(y_data).toarray()

x_train, x_test, y_train, y_test = train_test_split (x_data, y_data, train_size = 0.7, random_state=66)

#2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 54])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 7])

w = tf.compat.v1.Variable(tf.zeros([54,7]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1, 7]), name = 'bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1)) # categorical_crossentropy

optimizer =tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with  tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):
        _, cos_val, = sess.run([optimizer, loss], feed_dict = {x:x_train,y:y_train})
        if step % 200 ==0:
            print(step, cos_val)

    y_acc_test = sess.run(tf.argmax(y_test, 1))
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
    acc = accuracy_score(y_acc_test, predict)
    print("accuracy_score : ", acc)