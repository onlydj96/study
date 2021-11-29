import numpy as np
from sklearn import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

datasets = load_diabetes()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

model = Sequential()
model.add(Dense(1000, input_dim=10))
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1, verbose=3,
          validation_split=0.2)

loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
result = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, result)
print("r2 스코어 : ", r2)\
    
    
'''
loss :  2607.8134765625
r2 스코어 :  0.6017096800336375
'''

