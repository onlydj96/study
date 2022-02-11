from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(-1, 1)
print(x_data.shape)


x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.zeros([x_data.shape[1], 1]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1]), name='bias')  

#2. 모델
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
loss = -tf.reduce_mean(y*tf.math.log(hypothesis) + (1-y)*tf.math.log(1-hypothesis))  # binary_crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-6)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(3001):
    loss_val, hy_val, _= sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    
    if epoch % 200 == 0:
        print(epoch, '\t', loss_val)


#4. 평가, 예측
y_predict = tf.cast(hypothesis >= 0.5, dtype=tf.float32)  

# accuracy score
accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype=tf.float32))
pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_data, y:y_data})

print("Accuracy - \n" , acc)

sess.close()

# Accuracy - 0.9103691