import numpy as np

# 1차원 인덱싱
x = np.arange(7)
print(x) # [0 1 2 3 4 5 6]

# x의 4번째 숫자 찾기
print(x[3]) # 3

# x의 특정 숫자 바꾸기
'''
x[0] = 10
print(x) # [10, 1, 2, 3, 4, 5, 6]
'''

# 2차원 인덱싱
x = np.arange(1, 13)
x.shape = 3, 4
print(x) 