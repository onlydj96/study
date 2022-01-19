
# Pipeline을 여러 SearchCV와 엮어서 결과값을 도출

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np



#1. 데이터 분석
path = "D:/_data/kaggle/bike/"
train = pd.read_csv(path + "train.csv") # (10886, 12)

x = train.drop(columns=['datetime', 'casual', 'registered', 'count'], axis=1)
y = train['count']

y = np.log1p(y) 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

# Kfold
parameter = [
    {'rf__n_estimators' : [100, 200], 'rf__max_depth' : [6, 8, 10, 12], 
    'rf__min_samples_leaf' : [3, 4, 7, 10], 'rf__min_samples_split' : [3, 5, 10]},
]

#2. 모델구성
from sklearn.pipeline import Pipeline
pipe = Pipeline([('mm', MinMaxScaler()), ('rf', RandomForestRegressor())])

model = GridSearchCV(pipe, parameter, cv=5, verbose=1)

# 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time() - start
print("걸린 시간 : ", round(end, 4))
print("model.score : ", model.score(x_test, y_test)) 


'''
Fitting 5 folds for each of 36 candidates, totalling 180 fits
걸린 시간 :  122.5034
model.score :  0.3766707650223863
'''