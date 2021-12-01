import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


#1. 데이터 분석
path = "./_data/bike/"
train = pd.read_csv(path + "train.csv") # (10886, 12)
test_file = pd.read_csv(path + "test.csv") # (6493, 9)
submit_file = pd.read_csv(path + "sampleSubmission.csv") # (6493, 2)


# print(train.describe())
# train.info()
# print(train.columns)

# print(train.head()) # head : 앞의 값 5개
# print(train.tail()) # tail : 뒤의 값 5개

x = train.drop(columns=['datetime', 'casual', 'registered', 'count'], axis=1) # axis=0이 디폴드값이며, 디폴트일경우 컬럼단위가 아닌 행단위로 삭제됨
y = train['count']

test_file = test_file.drop(columns=['datetime'], axis=1) # 제출용 데이터 정제

'''
x의 값은 train데이터에서 4개의 열을 지운다
y의 값은 train데이터에 count만
'''


y = np.log1p(y) # 로그변환, 값중에 0은 로그를 사용할 수 없기때문에 1p(+1)을 사용하여 함수를 사용한다.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor="val_loss", patience=100, mode='min', verbose=1, restore_best_weights=True)

model.fit(x, y, epochs=1000, batch_size=16, verbose=1, validation_split=0.2, callbacks=[es])

#4. 결과
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)


from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_pred)
print("r2스코어", r2)

def RMSE(y_test, y_pred):     # tensorflow에서는 'rmse'를 지원하지 않기때문에 따로 정의해줘야한다.
    return np.sqrt(mean_squared_error(y_test, y_pred))   # sqrt은 SQuare RooT의 약자로서 제곱근을 의미한다
rmse = RMSE(y_test, y_pred) 
print("RMSE : ", rmse)

# plt.plot(y)
# plt.show

'''
평균(mean)은 데이터를 모두 더한 후 데이터의 갯수로 나눈 값이다. 중앙값(median)은 전체 데이터 중 가운데에 있는 수이다. 
데이터의 수가 짝수인 경우는 가장 가운데에 있는 두 수의 평균이 중앙값이다

loss :  1.6055853366851807
r2스코어 0.22331225173064462
RMSE :  1.2671170179833002
'''

###################################### 제출용 제작 ################################################# 
results = model.predict(test_file)

submit_file['count'] = results

# print(submit_file[:10])

submit_file.to_csv(path + "bikecount.csv", index=False)