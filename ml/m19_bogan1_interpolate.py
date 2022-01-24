
# 결측치 처리
# 1. 행 또는 열 삭제
# 2. 임의의 값
# Filling - 0, ffill, bfill, 중위값, 평균값...

# 3. 보간법 - interpolation 
# 4. 모델링 - predict
# 5. 부스팅게열 - 통상 결측치, 이상치에 대해 자유롭다. 믿거나 말거나


import pandas as pd
from datetime import datetime
import numpy as np

dates = ['01/24/2022', '01/25/2022', '01/26/2022', 
         '01/27/2022', '01/28/2022']

# pandas에서 datatime으로 변환하는 함수를 제공
dates = pd.to_datetime(dates) 
print(dates)

# Series는 백터를 제공
ts = pd.Series([2, np.nan, np.nan, 8, 10], index=dates)
print(ts)

# pandas에서 제공하는 Default interpolate는 선형회귀형 보간 모델
ts = ts.interpolate()
print(ts)