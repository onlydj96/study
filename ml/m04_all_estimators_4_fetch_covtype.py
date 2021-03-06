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
모델의 갯수:  41
AdaBoostClassifier 의 정답률:  0.5028613719095032
BaggingClassifier 의 정답률:  0.9613865390738621
BernoulliNB 의 정답률:  0.631833945767321
CalibratedClassifierCV 의 정답률:  0.7122621619063191
CategoricalNB 의 정답률:  0.6321437484402296
ComplementNB 의 정답률:  0.6225742880992745
DecisionTreeClassifier 의 정답률:  0.9396401125616378
DummyClassifier 의 정답률:  0.48625250638968015
ExtraTreeClassifier 의 정답률:  0.8489798025868524
ExtraTreesClassifier 의 정답률:  0.9541922325585398
GaussianNB 의 정답률:  0.09079800005163378
GradientBoostingClassifier 의 정답률:  0.773508429214392
HistGradientBoostingClassifier 의 정답률:  0.8105556655163808
KNeighborsClassifier 의 정답률:  0.9376263951877318
LinearDiscriminantAnalysis 의 정답률:  0.6797931206595355
LinearSVC 의 정답률:  0.7124170632427734
LogisticRegression 의 정답률:  0.7194220459024294
LogisticRegressionCV 의 정답률:  0.7246370575630577
MLPClassifier 의 정답률:  0.8393414972074731
MultinomialNB 의 정답률:  0.6410247583969433
NearestCentroid 의 정답률:  0.38585062347787924
PassiveAggressiveClassifier 의 정답률:  0.5204598848566733
Perceptron 의 정답률:  0.6010257910725196
QuadraticDiscriminantAnalysis 의 정답률:  0.08459334096365842
RandomForestClassifier 의 정답률:  0.9554400488799772
RidgeClassifier 의 정답률:  0.7008510967875184
RidgeClassifierCV 의 정답률:  0.7008424911577154
SGDClassifier 의 정답률:  0.7099902756383226
SVC 의 정답률:  0.773955921964149
'''