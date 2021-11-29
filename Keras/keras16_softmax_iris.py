import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

# print(datasets.DESCR) # x=(150, 4), y=(150,)

print(np.unique(y)) # y에 반환되는 값을 정리

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

'''
to_categorical : y의 종류가 몇 가지인지 확인하여 열을 종류의 개수로 바꾼다. 
즉, y = (150,)인데 (0, 1, 2)라는 세가지의 값만 반환된다면, y = (150, 3)이 된다.
'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=4))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3, activation='softmax'))  # 다중분류일 경우 output activation은 'softmax'이다


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 다중분류를 할 때, loss는 'categorical_crossentropy'이다.

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss[0])
print("accuracy : ", loss[1])
# result = model.predict(x_test[:7])
# print(result)

'''
loss :  0.07134769856929779
accuracy :  1.0
'''