
# 시각화

import tensorflow as tf, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(77)

#1. 데이터
x_train_data = [1,2,3]
y_train_data = [3,5,7]
x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])


w = tf.compat.v1.Variable(tf.random.normal([1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.random.normal([1]), dtype=tf.float32)

#2. 모델구성
hypothesis = x_train * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))  
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)   
train = optimizer.minimize(loss)    

#3-2. 훈련
with tf.compat.v1.Session() as sess:       
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    
    loss_val_list = []
    x_val_list = []
    
    for step in range(2000):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x_train:x_train_data, y_train:y_train_data})
        
        if (step+1) % 20 == 0:
            print(step+1, loss_val, w_val, b_val)
        
        loss_val_list.append(loss_val)
        x_val_list.append(w_val)

# 시각화
import matplotlib.pyplot as plt
plt.plot(loss_val_list[100:])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()