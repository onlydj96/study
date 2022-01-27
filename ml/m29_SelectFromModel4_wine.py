
from types import new_class
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import time
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = "D:/_data/winequality/"
datasets = pd.read_csv(path + 'winequality-white.csv', index_col = None, header = 0, sep=';') # 분리

datasets = datasets.values 

x = datasets[:,:11]  # 모든 행, 10번째까지
y = datasets[:, 11]  # 모든행, 11번째 열이 y 


# 나의 방식으로 칼럼 합치기 
# for i in range(len(y)):
#     if y[i] < 4:
#         y[i] = 4 
#     elif y[i] > 8:
#         y[i] = 8
# print(np.unique(y, return_counts=True))  # [4, 5, 6, 7, 8]

# 3개를 추출해서 새로운 y값을 출력
newlist = []
for i in y:
    if i <=4:
        newlist +=[0]
    elif i <=7:
        newlist +=[1]
    else:
        newlist +=[2]
newlist = np.array(newlist)
print(type(newlist))        
print(np.unique(newlist))


x_train, x_test, y_train, y_test = train_test_split(x, newlist, random_state=66, shuffle=True, train_size = 0.8, stratify = newlist)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 모델 구성

# RandomizedSearchCV를 통해서 최적의 파라미터를 찾아냄
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9,
              enable_categorical=False, eval_metric='merror', gamma=0,
              gpu_id=-1, importance_type=None, interaction_constraints='',
              learning_rate=0.3, max_delta_step=0, max_depth=5,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=8, num_parallel_tree=1,
              objective='multi:softprob', predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

#3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()-start
print("걸린 시간 : ", round(end, 2))

#4. 평가, 예측
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)
print(score)

# print(model.feature_importances_)
# # [0.07005408 0.10914277 0.07253726 0.091972   0.06962144 0.08850635
# #  0.06747309 0.07978433 0.07233866 0.06955209 0.20901796]

# print(np.sort(model.feature_importances_)) 
# # [0.06747309 0.06955209 0.06962144 0.07005408 0.07233866 0.07253726
# #  0.07978433 0.08850635 0.091972   0.10914277 0.20901796]


# fi = np.sort(model.feature_importances_)

# print("=====================================")

# for thresh in fi:
#     selection = SelectFromModel(model, threshold=thresh, prefit=True) 
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
#     print(select_x_train.shape, select_x_test.shape)
    
#     selection_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.9,
#               enable_categorical=False, eval_metric='merror', gamma=0,
#               gpu_id=-1, importance_type=None, interaction_constraints='',
#               learning_rate=0.3, max_delta_step=0, max_depth=5,
#               min_child_weight=1, monotone_constraints='()',
#               n_estimators=100, n_jobs=8, num_parallel_tree=1,
#               objective='multi:softprob', predictor='auto', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
#               tree_method='exact', validate_parameters=1, verbosity=None)
    
#     selection_model.fit(select_x_train, y_train)
#     y_predict = selection_model.predict(select_x_test)
#     score = accuracy_score(y_test, y_predict)
    
#     print("Thresh=%.3f, n=%d, score : %2f%%" 
#           %(thresh, select_x_train.shape[1], score*100))


'''
[0 1 2]
걸린 시간 :  0.25
0.9418367346938775
'''