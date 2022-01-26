
# pickle 함수를 이용해서 저장하기

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

# 훈련
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.025,
    max_depth=4,
    min_child_weight=1,
    subsample=1,
    colsample_bytree=1,
    reg_alpha=1,
    reg_lambda=0,
    n_jobs=-1
)

es_num = 10
start = time.time()
model.fit(x_train, y_train, verbose=0, 
          eval_set=[(x_test, y_test)],
          eval_metric='rmse',
          early_stopping_rounds=es_num,
          )
end = time.time()-start
print("걸린 시간 : ", round(end, 2))

score = model.score(x_test, y_test)
print("score : ", round(score, 3))

# 저장하기
import pickle
pickle.dump(model, open('D:/_save/m23_pickle_save.dat', 'wb'))
