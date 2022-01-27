


import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

#1. 데이터
x, y = fetch_covtype(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


# 모델 구성
# RandomizedSearchCV를 통해서 최적의 파라미터를 찾아냄
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9,
              enable_categorical=False, eval_metric='merror', gamma=0,
              gpu_id=-1, importance_type=None, interaction_constraints='',
              learning_rate=0.3, max_delta_step=0, max_depth=6,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=300, n_jobs=8, num_parallel_tree=1,
              objective='multi:softprob', predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

# 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()-start
print("걸린 시간 : ", round(end, 2))


# 측정

print(model.feature_importances_)
# [0.04817333 0.00526633 0.00341973 0.008279   0.00595322 0.0093423
#  0.00599185 0.00752786 0.00443197 0.00950437 0.04707012 0.02423839
#  0.0275787  0.10167357 0.00381011 0.03947886 0.01778033 0.0426585
#  0.0042983  0.00513966 0.00152276 0.008945   0.01061605 0.02014184
#  0.0150128  0.05553114 0.01191603 0.00403231 0.00075915 0.00581274
#  0.0132774  0.00526478 0.00724794 0.01147132 0.01613372 0.05370335
#  0.02676886 0.01466291 0.00375696 0.00621426 0.01950877 0.00289381
#  0.01834268 0.0166751  0.01679841 0.02899614 0.01456848 0.00725054
#  0.01688616 0.00336714 0.01645854 0.04254474 0.05145736 0.02984442]

print(np.sort(model.feature_importances_)) 
# [0.00075915 0.00152276 0.00289381 0.00336714 0.00341973 0.00375696
#  0.00381011 0.00403231 0.0042983  0.00443197 0.00513966 0.00526478
#  0.00526633 0.00581274 0.00595322 0.00599185 0.00621426 0.00724794
#  0.00725054 0.00752786 0.008279   0.008945   0.0093423  0.00950437
#  0.01061605 0.01147132 0.01191603 0.0132774  0.01456848 0.01466291
#  0.0150128  0.01613372 0.01645854 0.0166751  0.01679841 0.01688616
#  0.01778033 0.01834268 0.01950877 0.02014184 0.02423839 0.02676886
#  0.0275787  0.02899614 0.02984442 0.03947886 0.04254474 0.0426585
#  0.04707012 0.04817333 0.05145736 0.05370335 0.05553114 0.10167357]


fi = np.sort(model.feature_importances_)

print("=====================================")

for thresh in fi:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) 
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9,
              enable_categorical=False, eval_metric='merror', gamma=0,
              gpu_id=-1, importance_type=None, interaction_constraints='',
              learning_rate=0.3, max_delta_step=0, max_depth=5,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=8, num_parallel_tree=1,
              objective='multi:softprob', predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
    
    selection_model.fit(select_x_train, y_train)
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    
    print("Thresh=%.3f, n=%d, score : %2f%%" 
          %(thresh, select_x_train.shape[1], score*100))
