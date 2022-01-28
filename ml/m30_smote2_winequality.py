
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

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8, stratify = y)

print(np.unique(y_train, return_counts=True))  # [  16,  130, 1166, 1758,  704,  140,    4]


# Smote 적용(데이터 증폭하기)
smote = SMOTE(random_state=66, k_neighbors=1)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))  # [1758, 1758, 1758, 1758, 1758, 1758, 1758]

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
f1 = f1_score(y_test,pred,average='macro')
print(f1)

'''
1. SMOTE 전 : 0.41005452777318885
2. SMOTE 후 : 0.407321696642271
'''