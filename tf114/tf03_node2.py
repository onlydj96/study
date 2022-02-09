
# node에 사칙연산
 
import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

sess = tf.compat.v1.Session()

#1. 덧셈
# node3 = node1 + node2
node3 = tf.add(node1, node2)
print(sess.run(node3))

#2. 뺄샘
# node4 = node1 - node2
node4 = tf.subtract(node1, node2)
print(sess.run(node4))

#3. 곱셈 
# node5 = node1 * node2
node5 = tf.multiply(node1, node2)
print(sess.run(node5))

#4. 나눗셈
# node6 = node1 / node2
node6 = tf.divide(node1 , node2)
print(sess.run(node6))