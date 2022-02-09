
# placeholder의 역할

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # GPU, CPU관련 warning 잡아주기
tf.compat.v1.set_random_seed(77)

#1. 데이터
# x_train_data = [1,2,3]               # batch가 3인 상태와 같다.
# y_train_data = [1,2,3]
x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

# w = tf.compat.v1.Variable(1, dtype=tf.float32)
# b = tf.compat.v1.Variable(1, dtype=tf.float32)
w = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)

# sess.run(tf.global_variables_initializer())
# print(sess.run(w))  #이 한줄을 출력하기위해 위의 옵션이 필요하다...

#2. 모델구성
hypothesis = x_train * w + b           # hypothesis(가설) = y_predict

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))  

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)   

train = optimizer.minimize(loss)    


#3-2. 훈련
with tf.compat.v1.Session() as sess:       
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2000):
        # sess.run(train)     # 여기서 실행이 일어난다.
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x_train:[1,2,3], y_train:[1,2,3]})
        
        if (step+1) % 20 == 0:
            # print(f"{step+1}, {sess.run(loss)}, {sess.run(w)}, {sess.run(b)}")
            print(step+1, loss_val, w_val, b_val)