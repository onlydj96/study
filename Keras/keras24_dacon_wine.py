import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping


path = "../_data/dacon/wine/"
train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")

x = train.drop(columns=['quality'], axis=1)
y = train['quality']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(x['type'])
x['type'] = le.transform(x['type'])

le.fit(test_file['type'])
test_file['type'] = le.transform(test_file['type'])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

print(np.unique(y)) # [4, 5, 6, 7, 8]

print(x.shape, y.shape) # (3231, 13) (3231,)

model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, validation_split=0.2, batch_size=1, callbacks=[es])

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_pred = model.predict(x_test)

############################## 제출용 #####################################

results = model.predict(test_file)
submit_file['quality'] = results


submit_file.to_csv(path + "winequality.csv", index=False) # index=False