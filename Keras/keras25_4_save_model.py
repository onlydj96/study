import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

datasets = load_boston()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)

# 모델 구성
# model = Sequential()
# model.add(Dense(100, input_dim=13))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(3))
# model.add(Dense(1))

model = load_model("./_save/keras25_3_save_model.h5") 
'''
model.save를 model.fit다음에 저장하면 훈련된 weight값까지 저장한다.
'''

# model.summary()

# 컴파일

# model.save("./_save/keras25_3_save_model.h5")


# 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
result = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, result)
print("r2 스코어 : ", r2)

'''
save된 고정된 weight값
loss :  47.295372009277344
r2 스코어 :  0.43975258274331075
'''

