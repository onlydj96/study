
import tensorflow as tf
tf.compat.v1.set_random_seed(66)

x_data = [[73, 51, 65],                        # (5, 3)
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]

y_data = [[152], [185], [180], [205], [142]]   # (5, 1)