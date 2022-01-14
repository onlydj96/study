from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


#1. 데이터 분석
path = "../_data/kaggle/bike/"
train = pd.read_csv(path + "train.csv") # (10886, 12)
test_file = pd.read_csv(path + "test.csv") # (6493, 9)
submit_file = pd.read_csv(path + "sampleSubmission.csv") # (6493, 2)


# print(train.describe())
# train.info()
# print(train.columns)

# print(train.head()) # head : 앞의 값 5개
# print(train.tail()) # tail : 뒤의 값 5개

x = train.drop(columns=['datetime', 'casual', 'registered', 'count'], axis=1) # axis=0이 디폴드값이며, 디폴트일경우 컬럼단위가 아닌 행단위로 삭제됨
y = train['count']

test_file = test_file.drop(columns=['datetime'], axis=1) # 제출용 데이터 정제



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

#2. 모델구성
model = RandomForestRegressor()


#3. 컴파일


model.fit(x_train, y_train)

#4. 결과
y_pred = model.predict(x_test)


from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_pred)
print("r2스코어", r2)

def RMSE(y_test, y_pred):     # tensorflow에서는 'rmse'를 지원하지 않기때문에 따로 정의해줘야한다.
    return np.sqrt(mean_squared_error(y_test, y_pred))   # sqrt은 SQuare RooT의 약자로서 제곱근을 의미한다
rmse = RMSE(y_test, y_pred) 
print("RMSE : ", rmse)
