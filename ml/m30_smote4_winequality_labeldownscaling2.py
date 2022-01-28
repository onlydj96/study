
# 데이터를 증폭 후 성능비교

import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = "D:/_data/winequality/"
datasets = pd.read_csv(path + 'winequality-white.csv', index_col = None, header = 0, sep=';') # 분리

datasets = datasets.values
x = datasets[:,:11]
y = datasets[:, 11]

# 라벨 축소하기
# for i in range(len(y)):
#     if y[i] < 4:
#         y[i] = 4 
#     elif y[i] > 8:
#         y[i] = 8
# print(np.unique(y))  # [4, 5, 6, 7, 8]

for index, value in enumerate(y):
    if value == 9 :
        y[index] = 7
    elif value == 8 :
        y[index] = 7
    elif value ==  7:
        y[index] = 7
    elif value == 6 :
        y[index] = 6
    elif value == 5 :
        y[index] = 5
    elif value == 4 :
        y[index] = 5
    elif value == 3 :
        y[index] = 5
    else : 
        y[index] = 0

print(index, value)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8, stratify = y)

print(np.unique(y_train, return_counts=True)) # [ 146, 1166, 1758,  704,  144]

# Smote 적용(데이터 증폭하기)
smote = SMOTE(random_state=66, k_neighbors=1)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))  # [1758, 1758, 1758, 1758, 1758]

# 스케일러 적용
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
pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test,pred,average='macro')

print('acc : ', acc)
print('f1 : ', f1)
 

'''
1. SMOTE 전
acc :  0.65
f1 :  0.539374849915953

2. SMOTE 후
acc :  0.7142857142857143
f1 :  0.7113853377711319
'''
