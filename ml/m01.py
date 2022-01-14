from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


#2. 모델구성
from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(x_train, y_train)


#4. 평가, 예측

result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print('result : ', result)
print('acc : ', acc) 

