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


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)

model = SVC()

scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("acc : ", scores, "\n cross_val_score : ", round(np.mean(scores), 4))

model.fit(x_train, y_train)

pred = model.predict(x_test)
result = model.score(x_test, y_test)   # deep learning에 evaluate와 같다.
print(result)

'''
acc :  [0.95833333 1.         0.95833333 1.         0.875     ] 
 cross_val_score :  0.9583
 '''