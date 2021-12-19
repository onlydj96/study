import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

#1. 데이터 전처리
path = "../_data/dacon/heart_disease_pred/"
datasets = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit = pd.read_csv(path + "sample_submission.csv")

x = datasets.drop(columns=['id', 'target', 'restecg', 'chol', 'fbs'], axis=1)  # (151, 13)  ('restecg', 'chol', 'fbs')
y = datasets['target']   # (151,)
test_file = test_file.drop(columns=['id', 'restecg', 'chol', 'fbs'], axis=1)

# 결측치가 나온 행(index) 처리
x = x.drop(index=131)
y = y.drop(index=131)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
test_file = scaler.fit_transform(test_file)

#2. 모델구성
model = load_model("./_save/dacon_heart_disease_0.9714.h5") 

# 4. 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

# f1_score 계산
y_pred = model.predict(x_test)
y_pred = y_pred.round(0).astype(int)  # 0, 1로 떨어지게 만드는 수식
f1 = f1_score(y_test, y_pred)  # y_test와 y_pred를 비교하여 f1의 값을 도출하는 함수
print('f1 score :', f1)

################################ 제출 ###################################

result = model.predict(test_file)
result = result.round(0).astype(int)

submit['target'] = result
submit.to_csv(path + "heart_disease_pred.csv", index=False)