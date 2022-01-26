
# pickle 함수를 이용해서 데이터 저장하기

'''
pickle로 저장할 때 컬럼과 인덱스까지 그대로 가지고 저장한다.
'''

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, accuracy_score

import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (581012, 54) (581012,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 데이터 저장
import pickle
pickle.dump(datasets, open('D:/_save/m26_pickle_save_data.dat', 'wb'))

