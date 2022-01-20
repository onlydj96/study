from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


#2. 모델구성
# model = RandomForestClassifier()
# model = XGBClassifier()
model = GradientBoostingClassifier()

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측

result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print('acc : ', acc) 

print(model.feature_importances_)

'''
1. RandomForest
acc :  0.9666666666666667
[0.08700991 0.03042903 0.4507726  0.43178846]

2. XGboost 
acc :  0.9
[0.01835513 0.0256969  0.6204526  0.33549538]

3. GradientBoosting
acc :  0.9666666666666667
[0.00383314 0.0130144  0.26803841 0.71511405]
'''
