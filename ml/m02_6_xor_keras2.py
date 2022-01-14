import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron  
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

#2. 모델
model = Sequential()
model.add(Dense(100, input_dim=2, activation='sigmoid'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience=300, mode='max', restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=500, callbacks=[es])

#4. 평가, 예측
y_pred = model.predict(x_data)
print(x_data, "의 예측 결과 : ", y_pred)

results = model.evaluate(x_data, y_data)
print('metrics_acc : ', results[1])

# acc = accuracy_score(y_data, np.round(y_pred, 0))
