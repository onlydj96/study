
# lambda 함수 사용 및 iterator의 변화량 측정

import numpy as np

f = lambda x: x**2 - 4*x + 6

gradient = lambda x: 2*x - 4

x = 0.0   # 임의의 초기값
epochs = 8
learning_rate = 0.45

print("step\t x\t f(x)")
print("{:02d}\t {:6.5f}\t  {:6.5f}\t".format(0, x, f(x)))

for i in range(epochs):
    x = x - learning_rate * gradient(x)

    print("{:02d}\t {:6.5f}\t  {:6.5f}\t".format(i+1, x, f(x)))
