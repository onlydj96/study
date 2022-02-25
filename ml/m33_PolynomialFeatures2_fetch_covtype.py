from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

dataset = fetch_covtype()

x = dataset.data
y = dataset.target

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)
x = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.2)

model =  make_pipeline(StandardScaler(), CatBoostRegressor())
model.fit(x_train, y_train)

results = model.score(x_test, y_test)

print(results)

# 0.7555794581250297