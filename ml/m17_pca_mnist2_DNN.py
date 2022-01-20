# 실습 
# 아까 4가지로 모델을 만들고 비교하기

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)  # x_train, x_test를 행으로 합친다는 뜻

scaler = StandardScaler()
x_train = x_train.reshape(60000, -1)  # (60000, 784)
x_test = x_test.reshape(10000, -1)  # (10000, 784)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# pca를 통해 0.95 이상인 n_components가 몇개?

pca = PCA(n_components=154)  # 칼럼이 28*28개의 벡터로 압축이됨
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

print(x.shape)

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)   
# print(sum(pca_EVR)) 

# cumsum = np.cumsum(pca_EVR)  
# print(cumsum[0])

# 2. 모델구성

model=Sequential()
model.add(Dense(64, input_shape=(154,)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)

import time
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, callbacks=[es])
end = time.time()-start
print("걸린 시간 : ", round(end, 2))

#4. 예측
loss = model.evaluate(x_test, y_test)
print("loss, accuracy : ", loss)

acc = str(round(loss[1], 4))
model.save("./_save/dnn_mnist_{}.h5".format(acc))


'''
1. PCA 적용전
loss, accuracy :  [0.16708636283874512, 0.9592999815940857]

2. 0.95

3. 0.99

4. 0.999

5. 1.0
loss, accuracy :  [0.21637573838233948, 0.9509999752044678]
'''
