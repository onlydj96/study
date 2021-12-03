import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=1004)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

#2. 모델 구성
input1 = Input(shape=(10,))
hidden1 = Dense(100)(input1)
hidden2 = Dense(50, activation='relu')(hidden1)
hidden3 = Dense(10, activation='relu')(hidden2)
hidden4 = Dense(3, activation='relu')(hidden3)
output1 = Dense(1)(hidden4)
model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# model.add(Dense(100, input_dim=10))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(3, activation='relu'))
# model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

#4. 예측, 결과
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print("r2스코어", r2)
