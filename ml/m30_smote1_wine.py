
# 데이터 숫자 밸런스를 맞추기 위해서 특정 칼럼의 개수를 제거한다.
# SMOTE : synthetic minority oversampling technique

from catboost import train
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

datasets = load_wine()

x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (178, 13) (178,)
print(pd.Series(y).value_counts())

# 데이터 축소
x_new = x[:-30]
y_new = y[:-30]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66, stratify=y)


model = XGBClassifier(n_jobs=-1, eval_metric='merror',use_label_encoder=False)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("score : ", score)

'''
그냥 실행 - score :  0.9722222222222222

데이터 축소 - score :  0.9333333333333333
'''

print("==================SMOTE 적용====================")

smote = SMOTE(random_state=66)
x_train, y_train = smote.fit_resample(x_train, y_train)

model = XGBClassifier(n_jobs=-1, eval_metric='merror',use_label_encoder=False)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("score : ", score)

