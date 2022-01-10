import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler

#1. 데이터 전처리
path = "../_data/dacon/heart_disease_pred/"
datasets = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit = pd.read_csv(path + "sample_submission.csv")

x = datasets.drop(columns=['id', 'target', 'trestbps', 'restecg', 'chol', 'fbs'], axis=1)
y = datasets['target'] 
test_file = test_file.drop(columns=['id', 'trestbps', 'restecg', 'chol', 'fbs'], axis=1)   

# 결측치가 나온 행(index) 처리
x = x.drop(index=131)
y = y.drop(index=131)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=1)

scaler = MinMaxScaler()   #MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
test_file = scaler.fit_transform(test_file)

#2. 모델구성
model = Sequential()
model.add(Dense(512, input_dim=x.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, validation_split=0.25, batch_size=16, callbacks=[es])

# 4. 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

# f1_score 계산
y_pred = model.predict(x_test)
y_pred = y_pred.round(0).astype(int)  # 0, 1로 떨어지게 만드는 수식
f1 = f1_score(y_test, y_pred)  # y_test와 y_pred를 비교하여 f1의 값을 도출하는 함수
print('f1 score :', f1)


# 시각화 분석
import matplotlib.pyplot as plt
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'r--', label='training_loss')
plt.plot(epochs, val_loss, 'b:', label='training_val_loss')
plt.grid()
plt.legend()
plt.show()

################################ 제출 ###################################

result = model.predict(test_file)
result = result.round(0).astype(int)

save_f1 = str(round(f1, 4))
model.save("./_save/dacon_heart_disease_{}.h5".format(save_f1))

