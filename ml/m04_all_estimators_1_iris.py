from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_iris()
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
allAlgorithms = all_estimators(type_filter = 'classifier')  # classifier에 대한 모든 측정기
print("allAlgorithms: ", allAlgorithms)  # [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), ..]
print("모델의 갯수: ", len(allAlgorithms))  # 모델의 갯수:  41

for (name, algorithm) in allAlgorithms:  
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률: ', acc)
    except:                     # 에러나는 것 빼고 계속해라.
        # continue   
        print(name, "예외(에러남)")

'''
AdaBoostClassifier 의 정답률 :  0.36666666666666664
BaggingClassifier 의 정답률 :  0.36666666666666664
BernoulliNB 의 정답률 :  0.3
CalibratedClassifierCV 의 정답률 :  0.7
ComplementNB 의 정답률 :  0.7
DecisionTreeClassifier 의 정답률 :  0.36666666666666664
DummyClassifier 의 정답률 :  0.3
ExtraTreeClassifier 의 정답률 :  0.7
ExtraTreesClassifier 의 정답률 :  0.36666666666666664
GaussianNB 의 정답률 :  0.36666666666666664
GaussianProcessClassifier 의 정답률 :  0.36666666666666664
GradientBoostingClassifier 의 정답률 :  0.36666666666666664
HistGradientBoostingClassifier 의 정답률 :  0.36666666666666664
KNeighborsClassifier 의 정답률 :  0.36666666666666664
LabelPropagation 의 정답률 :  0.0
LabelSpreading 의 정답률 :  0.0
LinearDiscriminantAnalysis 의 정답률 :  0.7
LinearSVC 의 정답률 :  0.7
LogisticRegression 의 정답률 :  0.36666666666666664
LogisticRegressionCV 의 정답률 :  0.36666666666666664
MLPClassifier 의 정답률 :  0.36666666666666664
MultinomialNB 의 정답률 :  0.6333333333333333
NearestCentroid 의 정답률 :  0.36666666666666664
NuSVC 의 정답률 :  0.36666666666666664
PassiveAggressiveClassifier 의 정답률 :  0.7
Perceptron 의 정답률 :  0.7
QuadraticDiscriminantAnalysis 의 정답률 :  0.36666666666666664
RandomForestClassifier 의 정답률 :  0.36666666666666664
RidgeClassifier 의 정답률 :  0.36666666666666664
RidgeClassifierCV 의 정답률 :  0.6666666666666666
SGDClassifier 의 정답률 :  0.7
SVC 의 정답률 :  0.36666666666666664
'''