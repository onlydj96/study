import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터 전처리
path = "../_data/kaggle/titanic/"
train = pd.read_csv(path + "train.csv", index_col=0, header=0)  # (891, 12)
test = pd.read_csv(path + "test.csv")  # (418, 11)
submit = pd.read_csv(path + "gender_submission.csv")  # (418, 1)

x = train.drop(columns=['Survived', 'Cabin', 'Ticket', 'Name'], axis=1)
y = train["Survived"]
test_file = test.drop(columns=['Cabin', 'Ticket', 'Name'])

y = pd.get_dummies(y)

le = LabelEncoder()
le.fit(x['Sex'])
le.fit(['Embarked'])
x['Sex'] = le.transform(x['Sex'])
x['Embarked'] = le.transform(x['Embarked'])

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=7))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(patience=100, mode='auto', verbose=1, monitor='val_loss', restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es])

#4. 결과
loss = model.evaluate(x_test, y_test)
print('loss, accuracy : ', loss)

# 예측값 반환
result = model.predict(test_file)
# result_recover = np.argmax(result, axis=1).reshape(-1, 1) 
submit['Survived'] = result
# submit['Survived'] = result_recover
