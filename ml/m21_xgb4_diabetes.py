
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

import time

#1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 훈련
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    subsample=1,
    colsample_bytree=1,
    reg_alpha=1,
    reg_lambda=0,
    n_jobs=-1
)


start = time.time()
model.fit(x_train, y_train, verbose=1, 
          eval_set=[(x_test, y_test)],
          eval_metric='error',
          )
end = time.time()-start
print("걸린 시간 : ", round(end, 2))

score = model.score(x_test, y_test)
print("score : ", round(score, 3))

# y_pred = model.predict(x_test)
# r2 = r2_score(y_test, y_pred)
# print("r2 : ", r2)

'''
Default :  0.843
fetch_california_housing :  0.863
'''