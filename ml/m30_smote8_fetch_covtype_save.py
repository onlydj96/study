# Smote 비교
# 데이터를 증폭 후 성능비교

import numpy as np
import pandas as pd
import time
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
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

# Smote 적용(데이터 증폭하기)
smote = SMOTE(random_state=66, k_neighbors=2)
x_train, y_train = smote.fit_resample(x_train, y_train)

# Smote 데이터 저장
np.save("D:/_save/m30_xtrain_save.dat", arr = x_train)
np.save("D:/_save/m30_ytrain_save.dat", arr = y_train)

print(np.unique(y_train, return_counts=True))  # [1758, 1758, 1758, 1758, 1758]