import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.compat.v1.set_random_seed(66)

# random_normal에 들어가는 리스트는 차원을 의미한다. ex) [1] : dim=1
변수 = tf.compat.v1.Variable(tf.random.normal([1]), name='weight')

#1.
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수)
print("aaa : ", aaa)
sess.close()

#2. 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess)
print("bbb: ", bbb)
sess.close()

#3.
sess = tf.compat.v1.InteractiveSession() 
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
print("ccc : ", ccc)
sess.close()