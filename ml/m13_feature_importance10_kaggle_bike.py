
# feature_importance 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor



#1. 데이터 분석
path = "D:/_data/kaggle/bike/"
train = pd.read_csv(path + "train.csv") # (10886, 12)

x = train.drop(columns=['datetime', 'casual', 'registered', 'count'], axis=1)
y = train['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

#2. 모델구성
model1 = DecisionTreeRegressor()
model2 = RandomForestRegressor()
model3 = XGBRegressor()
model4 = GradientBoostingRegressor()

#3. 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)

#4. 스코어
score1 = model1.score(x_test, y_test)
score2 = model2.score(x_test, y_test)
score3 = model3.score(x_test, y_test)
score4 = model4.score(x_test, y_test)

import matplotlib.pyplot as plt

def plot_feature_importance_dataset(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_features), train.columns)
    plt.ylim(-1, n_features)
    


plt.subplot(2, 2, 1)
plot_feature_importance_dataset(model1)
plt.subplot(2, 2, 2)
plot_feature_importance_dataset(model2)
plt.subplot(2, 2, 3)
plot_feature_importance_dataset(model3)
plt.subplot(2, 2, 4)
plot_feature_importance_dataset(model4)

plt.show()