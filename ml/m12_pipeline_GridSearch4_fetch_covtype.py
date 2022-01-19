
# Pipeline을 여러 SearchCV와 엮어서 결과값을 도출

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

# parameter
parameter = [
    {'rf__n_estimators' : [100, 200], 'rf__max_depth' : [6, 8, 10, 12]},
    {'rf__max_depth' : [6, 8, 10, 12], 'rf__min_samples_leaf' : [3, 4, 7, 10]},
    {'rf__min_samples_leaf' : [3, 4, 7, 10], 'rf__min_samples_split' : [3, 5, 10]}
]

#2. 모델구성
from sklearn.pipeline import Pipeline
pipe = Pipeline([('mm', MinMaxScaler()), ('rf', RandomForestClassifier())])

model = GridSearchCV(pipe, parameter, cv=5, verbose=1)


#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()-start
print('걸린 시간 : ', round(end, 2))

#4. 평가, 예측
print("score : ", model.score(x_test, y_test))

'''
Fitting 5 folds for each of 36 candidates, totalling 180 fits
걸린 시간 :  17.56
score :  1.0
'''