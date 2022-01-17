import numpy as np
from sklearn.datasets import load_iris

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)

model = SVC()

scores = cross_val_score(model, x, y, cv=kfold)
print("acc : ", scores, "\n cross_val_score : ", round(np.mean(scores), 4))

'''
acc :  [0.96666667 0.96666667 1.         0.93333333 0.96666667] 
 cross_val_score :  0.9667
'''