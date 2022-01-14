from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

model = KNeighborsRegressor()

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측

result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print('acc : ', acc) 