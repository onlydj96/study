
# RandomizedSearchCV 사용하기
# Sklearn.model_selection에서 제공

from os import stat
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from sklearn.svm import SVC 

datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import KFold, cross_val_score, HalvingGridSearchCV


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)


parameter = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3, 4, 5]},                                    # 튜닝할 수 있는 경우의 수 : 12
    {"C":[1, 10, 100], 'kernel':["rbf"], 'gamma':[0.001, 0.0001]},                                        # 튜닝할 수 있는 경우의 수 : 06
    {"C":[1, 10, 100, 1000], "kernel":['sigmoid'], "gamma":[0.01, 0.001, 0.0001], "degree":[3, 4]}        # 튜닝할 수 있는 경우의 수 : 24
]                                                                                                         # 총 42개

# 모델 구성
import time
start = time.time()
model = HalvingGridSearchCV(SVC(), parameter, cv=kfold, verbose=1, refit=True, n_jobs=1) 
end = time.time() - start
print("걸린 시간 : ", round(end, 2))

'''
HalvingGridSearchCV : 연산 중 데이터 중 일부를 랜덤으로 추출 후 연산하여 성능이 좋은 상위 파라미터를 고른다. 이후 다시 연산하여 최적의 값을 뽑는다.

HalvingGridSearch와 RandomizedSearch의 차이점 : HalvingGrid는 데이터를 랜덤으로 추출한 것이고 Randomized는 파라미터 중 랜덤으로 추출하는 것이다.
보통은 Randomized를 주로 사용한다.
'''

# 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()-start
print("걸린 시간 : ", round(4, 4))


print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)

print("\nbest_score", model.best_score_)             # 훈련 시킨 것에서 최고 값
print("model.score : ", model.score(x_test, y_test)) # 테스트(예측)에서 최고 값

pred = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, pred))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적의 tuning acc : ", accuracy_score(y_test, y_pred_best))

