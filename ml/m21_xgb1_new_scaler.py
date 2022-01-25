
# 새로 import한 스케일러들의 의미파악

import pandas as pd
import time
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.model_selection import learning_curve, train_test_split, GridSearchCV

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score

import warnings
warnings.filterwarnings('ignore')

'''
1. QuantileTransformer : 지정된 분위수에 맞게 데이터를 변환한다. 기본 분위수는 1,000개이며 n_quantiles 매개변수에서 변경할 수 있음
2. PowerTransformer : 
'''

#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 훈련
model = XGBRegressor(
    n_estimators=2000,
    learning_rate = 0.1,
    max_depth=5,
    n_jobs=-1
)


start = time.time()
model.fit(x_train, y_train, verbose=1)
end = time.time()-start
print("걸린 시간 : ", round(end, 2))

score = model.score(x_test, y_test)
print("score : ", score)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

'''
1. boston : 0.9360888842883931

2. fetch_california_housing : 0.8531073618718206
'''