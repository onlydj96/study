
# Pipeline을 여러 SearchCV와 엮어서 결과값을 도출

from sklearn.datasets import load_iris
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

# parameter

'''
1. make_pipe에서 특정 머신의 parameter를 사용할 때에는 그 모델을 파라미터 앞에 붙인다
2. Pipeline에서 특정 머신의 parameter를 사용할 때에는 Pipeline에서 정의한 모델의 약자를 파라미터 앞에 붙인다
'''

#1. make_pipeline용
# parameter = [
#     {'randomforestclassifier__n_estimators' : [100, 200]},
#     {'randomforestclassifier__max_depth' : [6, 8, 10, 12]},
#     {'randomforestclassifier__min_samples_leaf' : [3, 4, 7, 10]},
#     {'randomforestclassifier__min_samples_split' : [3, 5, 10]}
# ]

#2. Pipeline용
parameter = [
    {'rf__n_estimators' : [100, 200], 'rf__max_depth' : [6, 8, 10, 12]},
    {'rf__max_depth' : [6, 8, 10, 12], 'rf__min_samples_leaf' : [3, 4, 7, 10]},
    {'rf__min_samples_leaf' : [3, 4, 7, 10], 'rf__min_samples_split' : [3, 5, 10]}
]

#2. 모델구성
from sklearn.pipeline import make_pipeline, Pipeline
pipe = Pipeline([('mm', MinMaxScaler()), ('rf', RandomForestClassifier())])
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model1 = GridSearchCV(pipe, parameter, cv=5, verbose=1)
model2 = RandomizedSearchCV(pipe, parameter, cv=5, verbose=1)
model3 = HalvingGridSearchCV(pipe, parameter, cv=5)

#3. 훈련
import time
start = time.time()
model1.fit(x_train, y_train)
end = time.time()-start
print('걸린 시간 : ', round(end, 2))

#4. 평가, 예측
print("score : ", model1.score(x_test, y_test))

'''
Fitting 5 folds for each of 13 candidates, totalling 65 fits
score :  0.9333333333333333
'''