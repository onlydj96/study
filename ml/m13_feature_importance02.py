
from sklearn.datasets import load_iris
import numpy as np

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x = np.delete(x, 0, axis=1)
x = np.delete(x, 1, axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


#2. 모델구성
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  # Regresison이라 부르지만, 분류모델!
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = DecisionTreeClassifier(max_depth=5)

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측

result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print('acc : ', acc)              # acc :  0.9

print(model.feature_importances_) # [0.04500046 0.95499954]
