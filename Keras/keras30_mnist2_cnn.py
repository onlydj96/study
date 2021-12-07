import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout



(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# x에 대한 전처리
x_train = x_train.reshape(60000, 28, 28, 1)  
x_test = x_test.reshape(10000, 28, 28, 1)
'''
이미지 데이터는 4차원(이미지 개수, 가로, 세로, 색)인데 shape을 확인해본 결과 '색' 부분이 안나타있다. 
그대로 x_train, x_test를 적용할 시 에러가 나기 때문에 x에 reshape을 적용하여 4차원 이미지 데이터로 만드는 전처리 과정이 필요하다. 
'''

# y에 대한 전처리(원핫인코딩)'

# print(np.unique(y_train, return_count=True)) # [0 1 2 3 4 5 6 7 8 9]
# return_count=True 함수는 전체 개수에서 np.unique의 각 컬럼의 개수가 나옴 

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)



# 모델 구성
model = Sequential()
model.add(Conv2D(4, kernel_size=(2,2), input_shape=(28, 28, 1)))   # 27, 27, 6
model.add(Conv2D(3, (3, 3), activation='relu'))   # 7, 7, 5
model.add(Conv2D(2, (2, 2), activation='relu'))   # 6, 6, 7
model.add(Flatten())
model.add(Dense(60))
model.add(Dropout(0.2))
model.add(Dense(30))
model.add(Dense(10, activation='softmax'))

# model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
model.fit(x_train, y_train, epochs=300, batch_size=1000, validation_split=0.2)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)