

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler


import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist

#1. 데이터 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# PCA
x = np.append(x_train, x_test, axis=0)  # x_train, x_test를 행으로 합친다는 뜻
x = x.reshape(70000, 28*28)
pca = PCA(n_components=154)  # 칼럼이 28*28개의 벡터로 압축이됨
x = pca.fit_transform(x)

# parameter
parameter = [
    {'xg__n_estimator':[100, 200, 300], 'xg__learning_rate':[0.1, 0.3, 0.001, 0.01],
     'xg__max_depth':[4, 5, 6]},
    {'xg__n_estimator':[90, 100, 110], 'xg__learning_rate':[0.1, 0.001, 0.01],
     'xg__max_depth':[4, 5, 6], 'xg__colsample_bytree':[0.6, 0.9, 1]},
    {'xg__n_estimator':[90, 100], 'xg__learning_rate':[0.1, 0.001, 0.5],
     'xg__max_depth':[4, 5, 6], 'xg__colsample_bytree':[0.6, 0.9, 1],
     'xg__colsample_bylevel':[0.6, 0.7, 0.9]}
]

#2. 모델 구성
from sklearn.pipeline import Pipeline
pipe = Pipeline([('mm', MinMaxScaler()), ('xg', XGBClassifier())])
model = GridSearchCV(pipe, parameter, cv=5, verbose=1, n_jobs=1)

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()-start
print('걸린 시간 : ', round(end, 2))

#4. 평가, 예측
print("score : ", model.score(x_test, y_test))

