import numpy as np

# *인덱싱*
# 1차원 인덱싱
# x = np.arange(7)
# print(x)

# print(x[3]) # x의 4번째 숫자 찾기
# print(x[0]) = 10 # x의 특정 숫자를 바꾸기

# 2차원 인덱싱
# x = np.arange(1, 13, 1)
# x.shape = 3, 4
# print(x)

# print(x[2, 3])

# *슬라이싱*
# # 1차원 슬라이싱
# x = np.arange(7)
# print(x)
# print(x[1:4]) # 1부터 3까지 추출
# print(x[1:]) # 1부터 끝까지 추출
# print(x[:4]) # 처음부터 3까지 추출
# print([::2]) # 마지막 "2"는 간격을 의미

# 2차원 슬라이싱
x = np.arange(1, 13, 1)
x.shape = 3, 4
print(x)

print(x[1:2,:2:3])
print(x[1:,:2])