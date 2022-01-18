
# 실습

# 모델 : RamdomForestClassifier


from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# 데이터 전처리
datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

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
model = GridSearchCV(RandomForestClassifier(), parameter, cv=kfold, verbose=1, refit=True)

# 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time() - start
print("걸린 시간 : ", round(end, 4))


print("최적의 매개변수 : ", model.best_estimator_)
print("model.score : ", model.score(x_test, y_test)) # 테스트(예측)에서 최고 값


'''
Fitting 5 folds for each of 35 candidates, totalling 175 fits
걸린 시간 :  21.3539
최적의 매개변수 :  RandomForestClassifier(max_depth=8, n_estimators=200)
model.score :  1.0
'''