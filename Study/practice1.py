import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

print(x)
x = x.T
print(x)