
# 결측치 채워넣기 - pandas 라이브러리를 이용

import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, np.nan, 8, 10],
                    [2, 4, np.nan, 8, np.nan],
                    [np.nan, 4, np.nan, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]])


# print(data.shape)  # (4, 5)

data = data.transpose()  # (4, 5) => (5, 4)
data.columns = ['a','b','c','d']  # column명 지정하기
# print(data)


# 1-1. 결측치 확인
print(data.isnull())  # data가 nan이면 True, 아니면 False를 반환
print(data.isnull().sum())  # nan값이 아닌 것의 합을 각 행마다 구해줌
print(data.info())


# 1-2 .결측치 삭제
print(data.dropna(axis=0))  # 결측치가 있는 행을 제거하고 남은 행들을 반환, axis=0이 디폴트값


# 2-1. 특정값 - 평균값
means = data.mean()
print(means)  # 각 컬럼별 평균값이 반환
data1 = data.fillna(means)  # 데이터를 평균값으로 채워서 반환
print(data1)

# 2-2.  특정값 - 중위값
meds = data.median()  # 중간에 위치한 값을 반환
print(meds)
data2 = data.fillna(meds)
print(data2)

# 2-3. 특정값 - ffill(front fill), bfill(back fill)
data3 = data.fillna(method='ffill')  # nan데이터를 앞에 있는 값으로 채워짐
print(data3)
data4 = data.fillna(method='bfill')  # nan데이터를 뒤에 있는 값으로 채워짐
print(data4)

# 2-4. 특정값 - 원하는 값 지정
data5 = data.fillna(7777777)
print(data5)