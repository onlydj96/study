import numpy as np

print("1차원 array")
array = np.array(range(10))
print(array)

# 1. 자료형을 출력하라
print(type(array))

# 2. 차원을 출력하라
print(array.ndim)

# 3. 모양을 출력하라
print(array.shape)

# 4. 크기를 출력하라
print(array.size)

# 5. dtype(data type)을 출력하라
print(array.dtype)

# 6. 인덱스 5의 요소를 출력하라
print(array[5])

# 7. 인덱스 3의 요소부터 인덱스 5 요소까지 출력하라
print(array[3:6])