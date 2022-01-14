from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, shuffle = True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter = 'classifier')
print("allAlgorithms: ", allAlgorithms)  
print("모델의 갯수: ", len(allAlgorithms)) 

for (name, algorithm) in allAlgorithms:  
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률: ', acc)
    except:                 
        continue   

'''

'''