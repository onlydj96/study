
# make_pipeline 사용

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

#Kfold
n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)

# parameter
parameter = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12], 'min_samples_split' : [2, 3, 5, 10]},
    {'max_depth' : [6, 8, 10, 12], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_leaf' : [3, 4, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4]}
]


#2. 모델구성
from sklearn.pipeline import make_pipeline, Pipeline
model = make_pipeline(MinMaxScaler(), GridSearchCV(RandomForestRegressor(), parameter, cv=kfold, verbose=1, refit=True, n_jobs=1))

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()-start
print("걸린 시간 : ", round(end, 2))

#4. 평가, 예측
print("score : ", model.score(x_test, y_test))

'''
Fitting 5 folds for each of 71 candidates, totalling 355 fits
걸린 시간 :  62.25
score :  0.9245499763330594
'''

#