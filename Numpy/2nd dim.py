import numpy as np

x = np.array(range(15))
x.shape = 3, 5
print(x) 

# 행은 인덱스 0-1까지, 열은 인덱스 0-1까지 출력 (0:n-1)
print(x[0:2,0:2]) 

# dtype을 str로 변경해서 출력
print(x.astype('str'))
