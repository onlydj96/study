#  1      357 
#  0      212

# 라벨 0을 112개 삭제해서 재구성

#  smote 추가해서 비교
# 데이터를 증폭 후 성능비교

import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target



# print(np.unique(y, return_counts=True))  #  (array([0, 1]), array([212, 357], dtype=int64))

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8, stratify = y)

# print(np.unique(y_train, return_counts=True)) # [ 146, 1166, 1758,  704,  144]

# # Smote 적용(데이터 증폭하기)
# smote = SMOTE(random_state=66, k_neighbors=1)
# x_train, y_train = smote.fit_resample(x_train, y_train)

# print(np.unique(y_train, return_counts=True))  # [1758, 1758, 1758, 1758, 1758]

# # 스케일러 적용
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# #2. 모델
# model = XGBClassifier(n_jobs = -1, eval_metric='error')

# #3. 훈련
# start = time.time()
# model.fit(x_train, y_train)
# end = time.time()-start
# print("걸린 시간 : ", round(end, 2))


# #4. 평가, 예측
# pred = model.predict(x_test)
# acc = accuracy_score(y_test, pred)
# f1 = f1_score(y_test,pred,average='macro')

# print('acc : ', acc)
# print('f1 : ', f1)
 

# '''
# 1. 
# Before
# acc :  0.956140350877193
# f1 :  0.9521289997480473

# Smote
# acc :  0.956140350877193
# f1 :  0.9521289997480473
# '''