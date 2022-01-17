from sklearn.utils import all_estimators
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_boston()
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
        continue   
    
'''모델의 갯수:  54
ARDRegression 의 정답률:  0.8119016106669672
AdaBoostRegressor 의 정답률:  0.9083322287330413
BaggingRegressor 의 정답률:  0.909125990194445
BayesianRidge 의 정답률:  0.8119880571377843
CCA 의 정답률:  0.7913477184424631
DecisionTreeRegressor 의 정답률:  0.7728310156770571
DummyRegressor 의 정답률:  -0.0005370164400797517
ElasticNet 의 정답률:  0.16201563080833725
ElasticNetCV 의 정답률:  0.8113737663385278
ExtraTreeRegressor 의 정답률:  0.7909191900504517
ExtraTreesRegressor 의 정답률:  0.9357776859916977
GammaRegressor 의 정답률:  0.1964792057029865
GaussianProcessRegressor 의 정답률:  -1.5789586750241171
GradientBoostingRegressor 의 정답률:  0.9463030835369497
HistGradientBoostingRegressor 의 정답률:  0.9323326124661162
HuberRegressor 의 정답률:  0.7958373281656513
KNeighborsRegressor 의 정답률:  0.8265307833211177
KernelRidge 의 정답률:  0.8032549585020776
Lars 의 정답률:  0.7746736096721598
LarsCV 의 정답률:  0.7981576314184019
Lasso 의 정답률:  0.242592140544296
LassoCV 의 정답률:  0.8125908596954046
LassoLars 의 정답률:  -0.0005370164400797517
LassoLarsCV 의 정답률:  0.8127604328474286
LassoLarsIC 의 정답률:  0.8131423868817644
LinearRegression 의 정답률:  0.8111288663608667
LinearSVR 의 정답률:  0.7078094667027885
MLPRegressor 의 정답률:  0.4399753007048969
NuSVR 의 정답률:  0.6254681434531
OrthogonalMatchingPursuit 의 정답률:  0.5827617571381449
OrthogonalMatchingPursuitCV 의 정답률:  0.78617447738729
PLSCanonical 의 정답률:  -2.2317079741425743
PLSRegression 의 정답률:  0.8027313142007888
PassiveAggressiveRegressor 의 정답률:  0.8057054589828977
PoissonRegressor 의 정답률:  0.6749600710148602
RANSACRegressor 의 정답률:  0.12075271863606263
RadiusNeighborsRegressor 의 정답률:  0.41191760158788593
RandomForestRegressor 의 정답률:  0.9240030797396604
Ridge 의 정답률:  0.8087497007195745
RidgeCV 의 정답률:  0.8116598578372464
SGDRegressor 의 정답률:  0.8267324341226274
SVR 의 정답률:  0.6597910766772523
TheilSenRegressor 의 정답률:  0.7820339440919427
TransformedTargetRegressor 의 정답률:  0.8111288663608667
TweedieRegressor 의 정답률:  0.19473445117356525
'''