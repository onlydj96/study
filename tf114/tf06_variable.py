
# 변수값 반환하기

import tensorflow as tf

sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer()  # 변수는 초기화되기 전에 사용할 수 없음, 초기화=사용가능하게 준비
sess.run(init)

print(sess.run(x))