
# pickle 함수를 이용해서 데이터 로드하기

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, accuracy_score

import time
import pickle
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = pickle.load(open('D:/_save/m26_pickle_save_data.dat', 'rb'))

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 데이터 저장
print(x_test.shape)