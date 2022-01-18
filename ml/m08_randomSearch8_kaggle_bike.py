
# 실습

# 모델 : RamdomForestClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
import pandas as pd



#1. 데이터 분석
path = "D:/_data/kaggle/bike/"
train = pd.read_csv(path + "train.csv") # (10886, 12)
test_file = pd.read_csv(path + "test.csv") # (6493, 9)
submit_file = pd.read_csv(path + "sampleSubmission.csv") # (6493, 2)


x = train.drop(columns=['datetime', 'casual', 'registered', 'count'], axis=1)
y = train['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

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
model = RandomizedSearchCV(RandomForestRegressor(), parameter, cv=kfold, verbose=1, refit=True, n_iter=20) 

# 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time() - start
print("걸린 시간 : ", round(end, 4))

print("최적의 매개변수 : ", model.best_estimator_)
print("model.score : ", model.score(x_test, y_test)) 


'''
Fitting 5 folds for each of 20 candidates, totalling 100 fits
걸린 시간 :  67.2838
최적의 매개변수 :  RandomForestRegressor(max_depth=10)
model.score :  0.3669301674839698
'''