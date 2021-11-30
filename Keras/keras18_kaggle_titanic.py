import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.metrics import accuracy

#1. 데이터 전처리
path = "./_data/titanic/"
train = pd.read_csv(path + "train.csv", index_col=0, header=0)
test = pd.read_csv(path + "test.csv")
gender_submission = pd.read_csv(path + "gender_submission.csv") #(891, 11)

print(train.shape)
print(test.shape) # (418, 10)
print(gender_submission.shape) # (418, 1)

# print(train, test, gender_submission) 
# train = (891, 12), test = (418, 11), gender_submission = (418, 2)



x_train = train.drop(columns=['Survived']).astype(int)
y_train = train["Survived"]
test_del_col = train.drop(columns = ['PassengerId'])

# print(x, y) # x = (891, 10), y = (891,)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#4. 결과
loss = model.evaluate(test, gender_submission)
print('loss : ', loss)