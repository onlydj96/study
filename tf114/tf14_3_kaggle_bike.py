
import pandas as pd
import numpy as np
import tensorflow as tf
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


x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random.normal([x_data.shape[1], 1]), name='weight')
b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias')

#2. 모델
hypothesis =  tf.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(100001):
    loss_val, w_val, _= sess.run([loss, loss, train], feed_dict={x:x_data, y:y_data})
    
    if epoch % 200 == 0:
        print(epoch, '\t', loss_val)


#4. 평가, 예측
predict = tf.matmul(x, w) + b   # predict = model.predict

y_predict = sess.run(predict, feed_dict={x:x_data, y:y_data})

from sklearn.metrics import r2_score
r2 = r2_score(y_data, y_predict)
print('r2스코어 : ', r2)

sess.close()

# r2스코어 :  0.3374947477075311