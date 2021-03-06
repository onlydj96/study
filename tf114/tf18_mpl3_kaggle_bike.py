
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

# 데이터 로드
path = "D:/_data/kaggle/bike/"
train = pd.read_csv(path + "train.csv") # (10886, 12)

# 데이터 형변환 pandas -> numpy
x_data = train.drop(['casual','registered','count'], axis=1)  
x_data['datetime'] = pd.to_datetime(x_data['datetime'])
x_data['year'] = x_data['datetime'].dt.year
x_data['month'] = x_data['datetime'].dt.month
x_data['day'] = x_data['datetime'].dt.day
x_data['hour'] = x_data['datetime'].dt.hour
x_data = x_data.drop('datetime', axis=1).to_numpy()
y_data = train['count'].to_numpy()

# reshape
y_data = y_data.reshape(-1, 1)
print(x_data.shape)


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=66)
x_train = np.array(x_train, dtype='float32')
x_test = np.array(x_test, dtype='float32')

x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random.normal([x_data.shape[1], 64]), name='weight')
b = tf.compat.v1.Variable(tf.random.normal([64]), name='bias')  

#2. 모델구성
hidden_layer1 = tf.sigmoid(tf.matmul(x, w) + b)

w1 = tf.compat.v1.Variable(tf.random.normal([64, 32], name='weight1'))
b1 = tf.compat.v1.Variable(tf.random.normal([32], name='bias1'))

hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal([32, 16], name='weight2'))
b2 = tf.compat.v1.Variable(tf.random.normal([16], name='bias2'))

hidden_layer3 = tf.nn.relu(tf.matmul(hidden_layer2, w2) + b2)

w3 = tf.compat.v1.Variable(tf.random.normal([16, 4], name='weight3'))
b3 = tf.compat.v1.Variable(tf.random.normal([4], name='bias3'))

hidden_layer4 = tf.nn.relu(tf.matmul(hidden_layer3, w3) + b3)

w4 = tf.compat.v1.Variable(tf.random.normal([4, 1], name='weight3'))
b4 = tf.compat.v1.Variable(tf.random.normal([1], name='bias3'))

output = tf.matmul(hidden_layer4, w4) + b4

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(output - y))    # mse
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(10001):
    loss_val, w_val, b_val, _= sess.run([loss, w, b, train], feed_dict={x:x_train, y:y_train})
    
    if epoch % 200 == 0:
        print(epoch, '\t', loss_val)


#4. 평가, 예측
y_predict = sess.run(output, feed_dict={x:x_test, y:y_test})

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

sess.close()

# r2스코어 :  0.537191227088543