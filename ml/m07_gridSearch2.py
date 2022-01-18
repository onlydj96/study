
# 하이퍼 파라미터 튜닝을 다양하게 적용하여 최적의 로스값을 찾기위한 모델링

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

'''
사이킷런에서는 분류 알고리즘이나 회귀 알고리즘에 사용되는 하이퍼파라미터를 순차적으로 입력해 학습을 하고 측정을 하면서 가장 좋은 파라미터를 알려준다. 
GridSearchCV가 없다면 max_depth 가 3일때 가장 최적의 스코어를 뽑아내는지 1일때 가장 최적인 스코어를  뽑아내는지 일일이 학습을 해야 한다. 
하지만 grid 파라미터 안에서 집합을 만들고 적용하면 최적화된 파라미터를 뽑아낼 수 있다.
'''

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)


parameter = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3, 4, 5]},                                   # 튜닝할 수 있는 경우의 수 : 12
    {"C":[1, 10, 100], 'kernel':["rbf"], 'gamma':[0.001, 0.0001]},                                       # 튜닝할 수 있는 경우의 수 : 6
    {"C":[1, 10, 100, 1000], "kernel":['sigmoid'], "gamma":[0.01, 0.001, 0.0001], "degree":[3, 4]}       # 튜닝할 수 있는 경우의 수 : 24
]                                                                                                        # 모든 경우의 수 : 42개

'''
내가 쓰고 싶은 parameter를 정해서 기입
'''

# 모델 구성
model = SVC(C=1, kernel='linear', degree=3)

# 훈련
model.fit(x_train, y_train)


# 평가 예측

# x_test = x_train   # 과적합 상황 보여주기
# y_test = y_train   # train 데이터로 best_estimator_ 로 예측 뒤 점수를 내면 best_score에서 나온다

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)

print("\nbest_score", model.best_score_)             # 훈련 시킨 것에서 최고 값
print("model.score : ", model.score(x_test, y_test)) # 테스트(예측)에서 최고 값

pred = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, pred))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적의 tuning acc : ", accuracy_score(y_test, y_pred_best))

'''
최적의 매개변수 :  SVC(C=1, kernel='linear')
최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

model.score :  0.9666666666666667
accuracy_score :  0.9666666666666667
'''

####################################################################################################################
# 가장 좋은 값 찾기


# print(model.cv_results_)
data = pd.DataFrame(model.cv_results_)
print(data)

bbb = data[['params', 'mean_test_score', 'rank_test_score', 
      'split0_test_score']]
# 'split1_test_score', 'split2_test_score', 
#   'split3_test_score3', 'split4_test_score']]

print(bbb)   # 가장 좋은 값부터 랭크 차례대로