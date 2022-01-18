
# 실습

# 모델 : RamdomForestClassifier


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

#Kfold
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


'''
min_samples_leaf : 말단 노드가 되기 위한 최소한의 샘플 데이터 수
min_samples_split : 디폴트는 2이다. 노드를 분할하기 위한 최소한의 샘플 데이터 수이다. 
예를 들면, 한번 나눴는데 한 집단에 2개의 데이터가 들어있다면 더 이상 나누지 않는 것이다. 만약에 1이라면 끝까지 계속 나누기 때문에 과적합의 가능성이 있다. 
'''


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


# 파라미터 조합 2개 이상

'''
Fitting 5 folds for each of 35 candidates, totalling 175 fits
걸린 시간 :  21.2799
최적의 매개변수 :  RandomForestClassifier(max_depth=6)
model.score :  0.9333333333333333
'''
