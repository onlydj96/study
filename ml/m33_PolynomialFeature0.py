from math import degrees
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(10).reshape(5, 2)

print(x)
print(x.shape)

pf = PolynomialFeatures(degree=2)

xp = pf.fit_transform(x)

print(xp)
print(xp.shape)

########################################
print("="*40)

x = np.arange(12).reshape(4, 3)

print(x)
print(x.shape)

pf = PolynomialFeatures(degree=2)

xp = pf.fit_transform(x)

print(xp)
print(xp.shape)

########################################
print("="*40)

x = np.arange(8).reshape(4, 2)

print(x)
print(x.shape)

pf = PolynomialFeatures(degree=3)

xp = pf.fit_transform(x)

print(xp)
print(xp.shape)
