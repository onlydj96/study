
# HalvingGridSearchCV 사용하기
# Sklearn.model_selection에서 제공

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import train_test_split, KFold, HalvingGridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


#1. 데이터 분석
path = "D:/_data/kaggle/bike/"
train = pd.read_csv(path + "train.csv") # (10886, 12)
test_file = pd.read_csv(path + "test.csv") # (6493, 9)
submit_file = pd.read_csv(path + "sampleSubmission.csv") # (6493, 2)


x = train.drop(columns=['datetime', 'casual', 'registered', 'count'], axis=1)
y = train['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Kfold
n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)

# parameter
parameter = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 4, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4]}
]


# 모델 구성
model = HalvingGridSearchCV(RandomForestRegressor(), parameter, cv=kfold, verbose=1, refit=True) 

# 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time() - start
print("걸린 시간 : ", round(end, 4))

print("최적의 매개변수 : ", model.best_estimator_)
print("model.score : ", model.score(x_test, y_test)) 


'''
n_iterations: 4
n_required_iterations: 4
n_possible_iterations: 4
min_resources_: 322
max_resources_: 8708
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 35
n_resources: 322
Fitting 5 folds for each of 35 candidates, totalling 175 fits
----------
iter: 1
n_candidates: 12
n_resources: 966
Fitting 5 folds for each of 12 candidates, totalling 60 fits
----------
iter: 2
n_candidates: 4
n_resources: 2898
Fitting 5 folds for each of 4 candidates, totalling 20 fits
----------
iter: 3
n_candidates: 2
n_resources: 8694
Fitting 5 folds for each of 2 candidates, totalling 10 fits
걸린 시간 :  45.2609
최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=10, min_samples_split=3)
model.score :  0.3586480859833474
'''