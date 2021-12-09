import numpy as np
import seaborn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
from tensorflow.keras.utils import to_categorical

#1. 데이터 정제
datasets = load_boston()
x = datasets.data
y = datasets.target

import pandas as pd
x_refine = pd.DataFrame(x, columns=datasets.feature_names)   # 데이터를 numpy형에서 pandas형으로 바꾼후, 각 column의 이름을 표기
x_refine['price'] = y  # pd.DataFrame에 넣어주기 위해서는 column에 이름이 필요
# print(x_refine.corr())   # corr() : 지정된 데이터의 상관관계를 수치화

# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(10,10))
# sns.heatmap(data=x_refine.corr(), square=True, annot=True, cbar=True)
# plt.show()

'''
1. 데이터를 이미지형(4차원) 데이터로 변환하기 이전에 y값과 비교하여 상관관계가 떨어지는 값을 찾는다.(열의 값이 짝수이면 생략해도 된다)
2. y값을 비교하기 위하여 y의 column의 이름을 지정해주고 pd.DataFrame에 넣어준다.

* 단순히 corr() 함수를 이용해서 상관관계가 가장 낮은 column을 찾을 수 있지만, matplotlib를 이용하여 데이터를 이미지화 시키면 더 찾기 쉽다. 
'''
x_refine = x_refine.drop(columns=['CHAS', 'price'], axis=1) # (506, 12)
x_refine = x_refine.to_numpy()

'''
y column과 비교했을 떄 상관관계가 낮은 행과 y를 제거한다.
최종 정제된 x데이터는 modeling과 compile을 위해서 다시 numpy형으로 바꾼다.
'''


# print(type(x_cnn))
# print(x_cnn)
# print(x_cnn.corr())


x_train, x_test, y_train, y_test = train_test_split(x_refine, y, train_size=0.8, random_state=1)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train).reshape(404, 4, 3, 1)
x_test = scaler.fit_transform(x_test).reshape(102, 4, 3, 1)



#2. 모델 구성

model = Sequential()
model.add(Conv2D(10, kernel_size=(2, 2), padding='same', input_shape=(4, 3, 1)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True)
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])


#4. 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_pred = model.predict(x_test)


from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_pred)
print("r2스코어", r2)

'''
loss :  29.36621856689453
r2스코어 0.7028534361664693
'''