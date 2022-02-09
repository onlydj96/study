
# tensorflow1의 구동원리 파악

import tensorflow as tf

print(tf.__version__)

hello = tf.constant("Hello World")

print(hello)   # Tensor("Const:0", shape=(), dtype=string)

sess = tf.compat.v1.Session()
print(sess.run(hello))  # b'Hello World'


'''
tf.constant : 상수(고정값)

tf.variable : 변수

tf.placeholder : 

sess run : tensorflow가 구동하는 매커니즘, sess run을 통과하지 않으면 값이 출력되지 않는다. tensorflow 버전 2.0에서 사라졌다.
'''