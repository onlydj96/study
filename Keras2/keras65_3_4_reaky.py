
# reaky relu

import numpy as np
import matplotlib.pyplot as plt

def reaky_relu(x):
    return np.maximum(0.01*x, x)
    
x = np.arange(-5, 5, 0.1)
y = reaky_relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# ReLU는 x < 0에서 모든 값이 0이지만, reaky relu는 0밑에 값에 작은 기울기를 부여함