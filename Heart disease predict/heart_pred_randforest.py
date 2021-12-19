import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler


#1. 데이터 전처리
path = "../_data/dacon/heart_disease_pred/"
datasets = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit = pd.read_csv(path + "sample_submission.csv")

x = datasets.drop(columns=['id', 'target', 'restecg', 'chol', 'fbs'], axis=1)  # (151, 13)  ('restecg', 'chol', 'fbs')
y = datasets['target']   # (151,)
test_file = test_file.drop(columns=['id', 'restecg', 'chol', 'fbs'])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


#2. 모델구성
from sklearn.ensemble import RandomForestClassifier
# 앙상블 - Voting 
rf = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=66)



# #3. 컴파일
rf.fit(x_train, y_train)


# #4. 예측, 평가
y_pred = rf.predict(x_test)

f1 = f1_score(y_test, y_pred)
print('f1 score :', f1)

result = rf.predict(test_file)

# import joblib
# joblib.dump(rf, '_save/heart_disease_pred_{0:.4f}.pkl'.format(f1))

# submit['target'] = result
# submit.to_csv(path + "heart_disease_pred.csv", index=False) 

# import matplotlib.pyplot as plt
