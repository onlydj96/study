import tensorflow as tf
tf.compat.v1.set_random_seed(66)

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

# Input Layer
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([2,5], name = 'weight1'))
b1 = tf.compat.v1.Variable(tf.random.normal([5], name = 'bias1'))      
# random_normal : 0~1 사이의 정규확률분포 값을 생성해주는 함수
# random_uniform : 0~1 사이의 균등확률분포 값을 생성해주는 함수

#2. 모델구성
hidden_layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(tf.random.normal([5, 1], name='weight2'))
b2 = tf.compat.v1.Variable(tf.random.normal([1], name='bias2'))

output = tf.sigmoid(tf.matmul(hidden_layer1, w2) + b2)

#3-1. 컴파일
loss = tf.reduce_mean(y*tf.log(output)+(1-y)*tf.log(1-output)) # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(10001):
    loss_val, hy_val, _ = sess.run([loss, output, train], feed_dict={x:x_data, y:y_data})
    
    if epoch % 200 == 0:
        print(epoch, 'loss:', loss_val)
    
#4. 예측
y_predict = tf.cast(output > 0.5, dtype = tf.float32)         
accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype = tf.float32))     
pred, acc = sess.run([y_predict, accuracy], feed_dict={x:x_data, y:y_data})    

print("="*22, "\nAccuracy: ", acc)

sess.close() 
# Accuracy:  0.75