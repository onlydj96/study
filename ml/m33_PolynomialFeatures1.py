from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

import warnings

from sympy import Li
warnings.filterwarnings('ignore')

dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.2)

model =  make_pipeline(StandardScaler(), LinearRegression())
model.fit(x_train, y_train)

print(model.score(x_test, y_test))
# 0.8111288663608656

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=7, scoring='r2')
print(scores)


from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)
xp = pf.fit_transform(x)
print(xp.shape)