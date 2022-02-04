
# KMeans

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

datasets = load_iris()
# x = datasets.data
y = datasets.target

irisDF = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(irisDF)

kmeans = KMeans(n_clusters=3, random_state=66)
kmeans.fit(irisDF)

'''
KMeans : x의 값들의 군집화를 비교하여 y값을 분류한다.
n_cluster : KMeans의 매개변수로서 원하는 군집화 정도를 표기한다.
'''

print(np.unique(np.sort(kmeans.labels_), return_counts=True))   # array([0, 1, 2]), array([62, 50, 38]
print(np.unique(y, return_counts=True))                         # array([0, 1, 2]), array([50, 50, 50]

print(accuracy_score(y, kmeans.labels_))                        # 0.8933333333333333

'''
원래의 y값과는 차이가 있음 (정확도가 완벽하지 않다)
'''
