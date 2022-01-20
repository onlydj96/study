# PCA 

'''
PCA(Principal component analysis) : 고차원의 데이터를 저차원의 데이터로 환원시키는 기법을 말한다.
'''

from unittest import result
import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape)  # (569, 30)

pca = PCA(n_components=10)  # 칼럼이 8개의 벡터로 압축이됨
x = pca.fit_transform(x)
print(x.shape)  # (569, 12)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)   # 압축된 개수대로 컬럼의 벡터화 수치를 나타냄
print(sum(pca_EVR))  # 상관계수의 합이 1이듯 압축된 크기의 합은 1이됨

cumsum = np.cumsum(pca_EVR)
print(cumsum)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

'''
1. PCA 안했을 때 
결과 :  0.9736842105263158

2. PCA 사용했을 때
결과 :  0.9473684210526315
'''