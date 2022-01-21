# PCA 

'''
LDA : PCA와 마찬가지로 축소 방법 중 하나이다.
LDA는 PCA와 유사하게 입력 데이터 세트를 저차원 공간으로 투영(project)해 차원을 축소하는 기법이지만, PCA와 다르게 LDA는 지도학습의 분류(Classification)에서 사용된다.
'''

from unittest import result
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

lda = LinearDiscriminantAnalysis()
pca = PCA(n_components=12)  # 칼럼이 8개의 벡터로 압축이됨

lda.fit(x, y)

x = lda.transform(x)

print(x.shape)

#2. 모델
from xgboost import XGBClassifier
model = XGBClassifier()

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            train_size=0.8, random_state=66)

#3. 훈련
model.fit(x_train, y_train)
result = model.score(x_test, y_test)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)


'''
1. 
결과 :  0.869392356479609

2.LDA
결과 :  0.7882498730669604
'''