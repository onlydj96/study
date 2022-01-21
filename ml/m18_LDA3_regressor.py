
# LDA

import ast
import numpy as np
from sklearn.datasets import load_boston, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import warnings
from sympy import Min
warnings.filterwarnings('ignore')


#1. 데이터 정제
# datasets = load_boston()
# datasets = load_diabetes()
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

y = y*1000
# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            train_size=0.8, random_state=66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print("LDA 전 : ", x_train.shape)

y_train = np.round(y_train)

# LDA
lda = LinearDiscriminantAnalysis(n_components=4)

lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)


#2. 모델
from xgboost import XGBRegressor
model = XGBRegressor()

#3. 훈련

model.fit(x_train, y_train, eval_metric='rmse')
result = model.score(x_test, y_test)


print("LDA 후 : ", x_train.shape)


#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)


'''
1. boston
LDA 전 :  (404, 13)
LDA 후 :  (404, 4)
결과 :  0.8049056695352406

2. diabetes
LDA 전 :  (353, 10)
LDA 후 :  (353, 5)
결과 :  0.34680264756770585

3. california_housing
LDA 전 :  (16512, 8)
LDA 후 :  (16512, 4)
결과 :  0.687345591319463
'''