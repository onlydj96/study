
# 머신러닝의 loss, verbose 확인하기
# fit의 파라미터에서 확인가능
# 훈련 과정을 matplotlib을 통해시각화

from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score

import time



#1. 데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 훈련
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=1,
    colsample_bytree=1,
    reg_alpha=1,            # 규제 L1
    reg_lambda=0,           # 규제 L2
    n_jobs=-1
)


start = time.time()
model.fit(x_train, y_train, verbose=1, 
          eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric='rmse')
end = time.time()-start
print("걸린 시간 : ", round(end, 2))

'''
fit에 내장된 파라미터
1. verbose : 머신러닝의 계산과정을 보여줌
2. eval_set : 머신러닝의 evaluate, 리스트 형태로 입력가능  ex) [(x_test, y_test), (x_train, y_train)] 
3. eval_metric : loss를 입력, rmse, mae, logloss, merror, error등등이 있다.
'''

score = model.score(x_test, y_test)
print("score : ", round(score, 3))

# y_pred = model.predict(x_test)
# r2 = r2_score(y_test, y_pred)
# print("r2 : ", r2)

'''
Default :  0.843
fetch_california_housing :  0.945
'''

print("===========================================")
hist = model.evals_result()   # history와 같은 기능
# print(type(hist.get('validation_0')))

# 결과값 시각화
import matplotlib.pyplot as plt

train, test = hist.get('validation_0').get('rmse'), hist.get('validation_1').get('rmse')

# epochs = range(1, len(hist.get('validation_0').get('rmse')) + 1)
    
plt.plot(train, 'r--', label="train loss")
plt.plot(test, 'b:', label="test loss")
plt.title('RMSE', fontsize=18)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid()
plt.legend()
plt.show()


