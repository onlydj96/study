from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


#2. 모델구성
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  # Regresison이라 부르지만, 분류모델!
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = Perceptron()

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
acc :  0.9333333333333333

2. LinearSVC
acc :  0.9666666666666667

3. SVC
acc :  0.9666666666666667

4. KNeighborClassfier
acc :  0.9666666666666667

5. LogisticRegression
acc :  1.0

6. DecisionTreeClassifier
acc :  0.9333333333333333

7. RandomForestClassifier
acc :  0.9666666666666667
'''
