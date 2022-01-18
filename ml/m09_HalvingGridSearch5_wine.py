
# HalvingGridSearch 사용하기
# Sklearn.model_selection에서 제공

from sklearn.datasets import load_wine
from sklearn.experimental import enable_halving_search_cv    # HalvingGridSearchCV를 가동시키기 위해서 import 해야함
from sklearn.model_selection import train_test_split, StratifiedKFold, HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier


# 데이터 전처리
datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


# Kfold
n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=66)

# parameter 
parameter = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 4, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4]}
]
# 모델 구성
model = HalvingGridSearchCV(RandomForestClassifier(), parameter, cv=kfold, verbose=1, refit=True, n_jobs=1) 


# 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()-start
print("걸린 시간 : ", round(end, 4))


# 측정
print("최적의 매개변수 : ", model.best_estimator_)
print("model.score : ", model.score(x_test, y_test)) 

'''
n_iterations: 2
n_required_iterations: 4
n_possible_iterations: 2
min_resources_: 30
max_resources_: 142
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 35
n_resources: 30
Fitting 5 folds for each of 35 candidates, totalling 175 fits
----------
iter: 1
n_candidates: 12
n_resources: 90
Fitting 5 folds for each of 12 candidates, totalling 60 fits
걸린 시간 :  31.4904
최적의 매개변수 :  RandomForestClassifier(max_depth=8)
'''