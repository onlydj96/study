
# RandomizedSearchCV 사용하기
# Sklearn.model_selection에서 제공


from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold


# 데이터 전처리
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

# Kfold
n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=66)

# parameter 정의
parameter = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 4, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4]}
]

# 모델 구성

model = RandomizedSearchCV(RandomForestClassifier(), parameter, cv=kfold, verbose=1, refit=True, n_iter=30) 

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
Fitting 5 folds for each of 30 candidates, totalling 150 fits
걸린 시간 :  20.1797
최적의 매개변수 :  RandomForestClassifier(max_depth=10)
model.score :  0.9736842105263158
'''