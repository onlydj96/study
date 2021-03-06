# boolean indexing : 배열의 각 요소의 선택여부를 boolean mask를 이용하여 지정하는 방식 즉, 조건에 맞는 데이터를 가져와 참, 거짓인지를 알려줌

import numpy as np

x = np.arange(7)

print(x)
print(x < 3) # x가 3보다 작은지 모든 리스트의 값에서 판별
print(x > 7)

# 조건에 맞는 x의 값을 반환하기
print(x[x < 3])
print(x[x % 2 == 0]) # 2로 나눈 나머지가 0인 값들을 반환