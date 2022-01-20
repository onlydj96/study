
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


#2. 모델구성
model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = XGBClassifier()
model4 = GradientBoostingClassifier()

#3. 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)

import matplotlib.pyplot as plt

def plot_feature_importance_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
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