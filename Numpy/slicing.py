import numpy as np

# 1차원 슬라이싱
x = np.arange(7)
print(x) # [0 1 2 3 4 5 6]

# 1-3까지 추출
print(x[1:4]) # [1 2 3]

# :n은 간격을 의미
print(x[0:3:2]) # [0 2]

# exercise
x = np.arange(1, 13, 1)
x.shape = 3, 4
print(x)

print(x[1:2,:2:3]) # [[5]]
print(x[1:,:2]) #[[ 5  6][ 9 10]]