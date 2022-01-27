
# 

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score, f1_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = "D:/_data/winequality/"
datasets = pd.read_csv(path + 'winequality-white.csv', index_col = None, header = 0, sep=';') # 분리


import matplotlib.pyplot as plt
grouped = datasets.groupby('quality')['quality'].count()

plt.bar(grouped.index, grouped)
plt.show()