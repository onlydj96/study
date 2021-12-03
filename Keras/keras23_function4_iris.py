import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# print(datasets.DESCR) # x=(150, 4), y=(150,)

print(np.unique(y)) # y에 반환되는 값을 정리

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
input1 = Input(shape=(4,))
hidden1 = Dense(100)(input1)
hidden2 = Dense(50, activation='relu')(hidden1)
hidden3 = Dense(10, activation='relu')(hidden2)
hidden4 = Dense(5, activation='relu')(hidden3)
output1 = Dense(3, activation='softmax')(hidden4)
model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# model.add(Dense(100, input_dim=4))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(5))
# model.add(Dense(3, activation='softmax'))  


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss[0])
print("accuracy : ", loss[1])
