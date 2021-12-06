import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import time 

#1. 데이터 정제
datasets = load_boston()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1004)


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(mode='min', patience=10, monitor='val_loss', verbose=1, restore_best_weights=True) 
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath='./_ModelCheckPoint/keras26_1_MCP.hdf5')

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1,
          validation_split=0.3, callbacks=[es, mcp])
end = time.time() - start

print("===================================================================")
print(hist.history['val_loss'])
print("===================================================================")


plt.figure(figsize=(9,5))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()

print("걸린 시간 : ", round(end, 2), '초')

model.save('./_save/keras26_1_save_model.h5')


loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
result = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, result)
print("r2 스코어 : ", r2)

