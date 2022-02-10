from sklearn.datasets import load_boston
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.set_random_seed(66)

datasets = load_boston()
x_data = datasets.data
y_data = datasets.target.reshape(506, -1)


x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random.normal([13, 1]), name='weight')
b = tf.compat.v1.Variable(tf.random.normal([1]), name='bias')  

#2. 모델
hypothesis =  tf.matmul(x, w) + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-6)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(500001):
    loss_val, w_val, _= sess.run([loss, loss, train], feed_dict={x:x_data, y:y_data})
    
    if epoch % 200 == 0:
        print(epoch, '\t', loss_val, '\t', w_val)



#4. 평가, 예측
predict = tf.matmul(x, w) + b   # predict = model.predict

y_predict = sess.run(predict, feed_dict={x:x_data, y:y_data})

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(y_data, y_predict)
print('r2스코어 : ', r2)

sess.close()

# r2스코어 :  0.6507726037467685