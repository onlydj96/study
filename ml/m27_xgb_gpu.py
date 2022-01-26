
# 머신러닝 GPU로 계산하기

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

#2. 훈련 가중치 불러오기
model = XGBClassifier(
    n_estimators=10000,
    learning_rate = 0.001,
    max_depth=20,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    gpu_id=0
)

start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_metric='mlogloss',
          eval_set=[(x_test, y_test)],
          early_stopping_rounds=20)
end = time.time()-start
print("걸린 시간 : ", round(end, 2))

score = model.score(x_test, y_test)
print("score : ", score)

'''
걸린 시간 :  11.29
score :  0.8154608745041005
'''