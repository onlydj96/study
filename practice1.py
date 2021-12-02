import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

path = "./_data/telecom_customers/"
train = pd.read_csv(path + "churn-bigml-80.csv")
test_file = pd.read_csv(path + "churn-bigml-80.csv")\
    


# train = pd.get_dummies(train['International plan', 'Voice mail plan', 'Churn'])
# train = pd.get_dummies(train['State'], prefix = 'State')

# print(train.shape) # (2666, 20)
# print(test_file.shape) # (2666, 20)

from sklearn.preprocessing import OneHotEncoder

x = train.drop(columns=['Churn'], axis=1)
y = train['Churn']
print(np.unique(y))
ohe = OneHotEncoder(sparse=False)

y_ohe = ohe.fit(x[['State']])
y = ohe.transform(x[['State']])



# test_file = test_file.drop(columns=['Churn'], axis=1)

# print(x.shape, y.shape) # (2666, 19), (2666,)

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

# # scaler = MinMaxScaler()
# # scaler.fit(x_train)
# # x_train = scaler.transform(x_train)
# # x_test = scaler.transform(x_test)

# # 모델
# model = Sequential()
# model.add(Dense(100, input_dim=19))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# # 컴파일
# model.compile(loss='binary_crossentropy', optimizer='adam')
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
# model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es])

# # 예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# y_pred = model.predict(x_test)
