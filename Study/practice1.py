import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.engine import sequential
from tensorflow.python.ops.array_ops import sequence_mask

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([102, 107, 110, 111, 115, 119, 124, 129, 130, 137])

x_train = x[:7]
y_train = y[:7]

x_test = x[7:]
y_test = y[7:]

print(x.shape)