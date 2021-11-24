import numpy as np
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# train과 test를 8:2으로 분류하시오.
x_train = x[0:8]
x_test = x[8:]
y_train = y[0:8]
y_test = y[8:]

print(x_train)
print(x)

print(x_train.ndim)