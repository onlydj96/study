# Smote 비교
# 데이터를 증폭 후 성능비교

import numpy as np
import pandas as pd
import time
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(np.unique(y, return_counts=True))  #  (array([0, 1]), array([212, 357], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8, stratify = y)

print(np.unique(y_train, return_counts=True)) # [ 146, 1166, 1758,  704,  144]

# SMOTE 데이터 로드
x_train = np.load("D:/_save/m30_xtrain_pickle_save.dat", allow_pickle=True)
y_train = np.load("D:/_save/m30_ytrain_pickle_save.dat", allow_pickle=True)

print(np.unique(y_train, return_counts=True))

# 스케일러 적용
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameter = [
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01, 0.5],
     'max_depth':[4, 5, 6], 'colsample_bytree':[0.6, 0.9, 1], 'eval_metric':['merror']}]

#2. 모델
model = model = GridSearchCV(XGBClassifier(), parameter, cv=5, verbose=1, refit=True)

model.fit(x_train, y_train)
# #4. 평가, 예측
pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test,pred,average='macro')

print('acc : ', acc)
print('f1 : ', f1)
 

# '''
# 1. 
# Before
# 걸린 시간 :  22.11
# acc :  0.9276008364672168
# f1 :  0.9203903770704602

# Smote
# acc :  0.956140350877193
# f1 :  0.9521289997480473
# '''