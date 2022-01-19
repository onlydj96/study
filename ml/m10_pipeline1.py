
# make_pipeline 사용

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

#2. 모델구성
from sklearn.pipeline import make_pipeline, Pipeline
model = make_pipeline(MinMaxScaler(), SVC())


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("score : ", model.score(x_test, y_test))

'''
score :  1.0
'''