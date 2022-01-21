
# LDA

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')


#1. 데이터 정제
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, -1)  # (60000, 784)
x_test = x_test.reshape(10000, -1)  # (10000, 784)

print("LDA 전 : ", x_train.shape)

parameter = [
    {'xg__n_estimators':[100, 200, 300], 'xg__learning_rate':[0.1, 0.3, 0.001, 0.01],
     'xg__max_depth':[4, 5, 6], 'xg__colsample_bytree':[0.6, 0.9, 1], 'xg__eval_metric':['merror']}]

#2. 모델 구성
from sklearn.pipeline import Pipeline
pipe = Pipeline([('mm', MinMaxScaler()), ('LDS', LinearDiscriminantAnalysis()), ('xg', XGBClassifier())])
model = GridSearchCV(pipe, parameter, cv=5, verbose=1, n_jobs=1)

#3. 훈련
model.fit(x_train, y_train)
result = model.score(x_test, y_test)


print("LDA 후 : ", x_train.shape)


#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)


'''
1. mnist
LDA 전 :  (60000, 784)
LDA 후 :  (60000, 4)
결과 :  0.8282
'''