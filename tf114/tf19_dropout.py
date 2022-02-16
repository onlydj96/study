import tensorflow as tf

dropout = tf.nn.dropout(x, keep_prob=0.7)  # keep_prob= 그대로 가져갈 노드수의 비율 