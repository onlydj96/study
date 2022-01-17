import numpy as np
import pandas as pd
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#1. 데이터 분석
path = "D:/_data/kaggle/bike/"
train = pd.read_csv(path + "train.csv") # (10886, 12)
test_file = pd.read_csv(path + "test.csv") # (6493, 9)
submit_file = pd.read_csv(path + "sampleSubmission.csv") # (6493, 2)


x = train.drop(columns=['datetime', 'casual', 'registered', 'count'], axis=1)
y = train['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

#2. 모델구성
allAlgorithms = all_estimators(type_filter = 'regressor')
print("allAlgorithms: ", allAlgorithms)  
print("모델의 갯수: ", len(allAlgorithms)) 

for (name, algorithm) in allAlgorithms:  
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = r2_score(y_test, y_predict)
        print(name, '의 정답률: ', acc)
    except:                 
        print(name, "예외(에러남)")

