
# 이렇게 하면 tensorflow2.7.0 서도 구동 가능

import tensorflow as tf

print(tf.__version__)
print(tf.executing_eagerly())  # False


# 즉시 실행 모드
tf.compat.v1.disable_eager_execution()  # 꺼!!

print(tf.executing_eagerly())  # False


hello = tf.constant("Hello World")

sess = tf.compat.v1.Session()
print(sess.run(hello))