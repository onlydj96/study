from sklearn.utils import all_estimators
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_diabetes()
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
    
'''
모델의 갯수:  54
ARDRegression 의 정답률:  0.4987482890562194
AdaBoostRegressor 의 정답률:  0.3454001391389324
BaggingRegressor 의 정답률:  0.29899245247931183
BayesianRidge 의 정답률:  0.501436686384745
CCA 의 정답률:  0.48696409064967605
DecisionTreeRegressor 의 정답률:  -0.20395538714987937
DummyRegressor 의 정답률:  -0.00015425885559339214
ElasticNet 의 정답률:  0.11987522766332959
ElasticNetCV 의 정답률:  0.489413697359085
ExtraTreeRegressor 의 정답률:  -0.3296120554913953
ExtraTreesRegressor 의 정답률:  0.38667468727100907
GammaRegressor 의 정답률:  0.07219655012236648
GaussianProcessRegressor 의 정답률:  -7.547010959397001
GradientBoostingRegressor 의 정답률:  0.3876243989858523
HistGradientBoostingRegressor 의 정답률:  0.28899497703380905
HuberRegressor 의 정답률:  0.5068530796935881
KNeighborsRegressor 의 정답률:  0.3741821819765594
KernelRidge 의 정답률:  0.48022687224693394
Lars 의 정답률:  0.49198665214641624
LarsCV 의 정답률:  0.5010892359535755
Lasso 의 정답률:  0.4643075327668871
LassoCV 의 정답률:  0.4992382182931273
LassoLars 의 정답률:  0.36543887418957943
LassoLarsCV 의 정답률:  0.495194279067825
LassoLarsIC 의 정답률:  0.49940515175310696
LinearRegression 의 정답률:  0.5063891053505036
LinearSVR 의 정답률:  0.14937513640686884
MLPRegressor 의 정답률:  -0.5473295258107862
NuSVR 의 정답률:  0.12527149380257419
OrthogonalMatchingPursuit 의 정답률:  0.3293449115305739
OrthogonalMatchingPursuitCV 의 정답률:  0.44354253337919736
PLSCanonical 의 정답률:  -0.9750792277922924
PLSRegression 의 정답률:  0.4766139460349792
PassiveAggressiveRegressor 의 정답률:  0.4683350647204231
PoissonRegressor 의 정답률:  0.48232318748923586
RANSACRegressor 의 정답률:  -0.3236698562209006
RadiusNeighborsRegressor 의 정답률:  0.14407236562185122
RandomForestRegressor 의 정답률:  0.3759962130314948
Ridge 의 정답률:  0.49950383964954104
RidgeCV 의 정답률:  0.49950383964954104
SGDRegressor 의 정답률:  0.49550921840004314
SVR 의 정답률:  0.12343791188320263
TheilSenRegressor 의 정답률:  0.5089816804191911
TransformedTargetRegressor 의 정답률:  0.5063891053505036
TweedieRegressor 의 정답률:  0.07335459385974397
'''