# Fancy indexing : 배열이 각 요소 선택을 Index 배열을 전달하여 지정하는 방식, 즉 Index에 어떤 값이 있는지를 물어보는 것

import numpy as np

# x = np.arange(7)
# print(x)
# print(x[[1, 3, 5]])

x = np.arange(1, 13, 1).reshape(3, 4)
print(x)

print(x[[0, 2]]) # 0번째 행과 2번째 행을 추출
