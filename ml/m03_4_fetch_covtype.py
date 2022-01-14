from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

model = KNeighborsClassifier()

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측

result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print('acc : ', acc) 


'''
분류기 종류별 스코어 비교
1. Perceptron
acc :  0.6388888888888888  ? 

2. LinearSVC
acc :  0.7222222222222222 ? 

3. SVC
acc :  0.6944444444444444 ? 

4. KNeighborsClassifier
0.5900872726222293

5. LogisticRegression
acc :  0.6199753878987634

6. DecisionTreeClassifier
acc :  0.9395798731530167

7. RandomForestClassifier
acc :  0.9553367813223411
'''
