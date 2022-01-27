
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score, f1_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = "D:/_data/winequality/"
datasets = pd.read_csv(path + 'winequality-white.csv', index_col = None, header = 0, sep=';') # 분리

datasets = datasets.values #  pandas --> numpy로 바꿔주기
#print(type(datasets)) # <class 'numpy.ndarray'>

x = datasets[:,:11]  # 모든 행, 10번째까지
y = datasets[:, 11]  # 모든행, 11번째 열이 y 

print(np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8, stratify = y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(n_jobs = -1, eval_metric='merror')
#3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()-start
print("걸린 시간 : ", round(end, 2))

#4. 평가, 예측
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)
print('model.score: ', score)
print('acc_score: ', accuracy_score(y_test, y_predict))
# print('f1_score: ', f1_score(y_test, y_predict, average='macro'))   


''' 
model.score:  0.6591836734693878
acc_score:  0.6591836734693878
'''