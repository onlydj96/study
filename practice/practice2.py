import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.python.keras.callbacks import History

#1. 데이터 전처리
path = "../_data/dacon/heart_disease_pred/"
datasets = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit = pd.read_csv(path + "sample_submission.csv")

x = datasets.drop(columns=['id', 'target', 'restecg', 'chol', 'fbs'], axis=1)  # (151, 13)  ('restecg', 'chol', 'fbs')
y = datasets['target']   # (151,)
test_file = test_file.drop(columns=['id', 'restecg', 'chol', 'fbs'], axis=1)

x = x.drop(index=131)
y = y.drop(index=131)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
test_file = scaler.fit_transform(test_file)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=x.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, validation_split=0.2, batch_size=1, callbacks=[es])

# 4. 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = model.predict(x_test)
y_pred = y_pred.round(0).astype(int)

f1 = f1_score(y_test, y_pred)
print('f1 score :', f1)

result = model.predict(test_file)
result = result.round(0).astype(int)

save_f1 = str(round(f1, 4))
model.save("./_save/dacon_heart_disease_{}.h5".format(save_f1))

submit['target'] = result
submit.to_csv(path + "heart_disease_pred.csv", index=False) 

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,9))
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.show()