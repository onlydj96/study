
import tensorflow as tf
import numpy as np
import os
tf.compat.v1.set_random_seed(66)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#1. 데이터
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]  # (6, 2)
y_data = [[0], [0], [0], [1], [1], [1]]                    # (6, 1)

#2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])         
                                                                     # 행열 연산은 앞의 열과 뒤의 행의 shape이 맞아야 함 w의 행은 x의 열의 갯수 와 동일한 shape이여야 함  
w = tf.compat.v1.Variable(tf.random.normal([2,1]), name='weight')    # y = x * w  ;  (5, 1) = (5, 3) * (? * ?)  => (3, 1)    
b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias')        # bias는 덧셈이므로 shape변화 없음

#2. 모델
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))  # binary_crossentropy
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.04)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(2001):
    loss_val, hy_val, _= sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    
    if epoch % 200 == 0:
        print(epoch, '\t', loss_val, '\t', hy_val)



#4. 평가, 예측
y_predict = tf.cast(hypothesis >= 0.5, dtype=tf.float32)   # 텐서를 새로운 형태로 캐스팅하는데 사용한다. 부동소수점형에서 정수형으로 바꾼 경우 소수점 버린을 한다. 
                                                           # Boolean형태인 경우 True이면 1, False이면 0을 출력한다.
                                                        
# accuracy score
accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype=tf.float32))
pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_data, y:y_data})

print("예측값 - \n" , hy_val)
print("예측결과 - \n" , pred)
print("Accuracy - \n" , acc)

sess.close()