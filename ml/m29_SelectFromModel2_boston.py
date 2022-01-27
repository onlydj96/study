


import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel

#1. 데이터
x, y = load_boston(return_X_y=True)
x = np.delete(x, (3, 1, 6, 11), axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBRegressor(n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
print("score : ", score)


# feature importance에서의 값이 낮은 것을 차례대로 분류하여 수치가 높은 데이터로만 모델링하는 for문

print(model.feature_importances_)
# [0.01447933 0.00363372 0.01479118 0.00134153 0.06949984 0.30128664
#  0.01220458 0.05182539 0.0175432  0.03041654 0.04246344 0.01203114
#  0.4284835 ]

print(np.sort(model.feature_importances_)) 
# [0.00134153 0.00363372 0.01203114 0.01220458 0.01447933 0.01479118
#  0.0175432  0.03041654 0.04246344 0.05182539 0.06949984 0.30128664
#  0.4284835 ]


fi = np.sort(model.feature_importances_)

print("=====================================")

for thresh in fi:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) 
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    
    print("Thresh=%.3f, n=%d, R2 : %2f%%" 
          %(thresh, select_x_train.shape[1], score*100))
