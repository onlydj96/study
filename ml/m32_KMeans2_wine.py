
# KMeans

from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

datasets = load_wine()
# x = datasets.data
y = datasets.target

wineDF = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(wineDF)

kmeans = KMeans(n_clusters=3, random_state=66)
kmeans.fit(wineDF)


print(np.unique(np.sort(kmeans.labels_), return_counts=True))   # array([0, 1, 2]), array([62, 50, 38]
print(np.unique(y, return_counts=True))                         # array([0, 1, 2]), array([50, 50, 50]
print(accuracy_score(y, np.sort(kmeans.labels_)))               # 0.8651685393258427
