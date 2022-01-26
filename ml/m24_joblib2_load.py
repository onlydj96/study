
# pickle 함수를 이용해서 로드하기

from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, accuracy_score

import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 훈련 가중치 불러오기
import joblib
model = joblib.load("D:/_save/m24_joblib_save.dat")


#3. 평가
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("score : ", round(r2, 3))

print("==============================")
hist = model.evals_result()
print(len(hist.get('validation_0').get('rmse')))