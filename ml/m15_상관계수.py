from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# print(datasets.feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

df = pd.DataFrame(x, columns=datasets.feature_names)
# df = pd.DataFrame(x, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
df['Target(Y)'] = y
print(df)

print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()