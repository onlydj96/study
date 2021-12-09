from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar100.load_data()


print(x_train.shape)

scaler = MinMaxScaler()
x_train = x_train.reshape(50000, -1)  # (50000, 3072)
x_test = x_test.reshape(10000, -1)  # (10000, 3072)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성

model=Sequential()
model.add(Dense(64, input_shape=(3072, )))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.3, callbacks=[es])


#4. 예측
loss = model.evaluate(x_test, y_test)
print("loss, accuracy : ", loss)

acc = str(round(loss[1], 4))
model.save("./_save/dnn_cifar10_{}.h5".format(acc))

'''
loss, accuracy :  [4.605363845825195, 0.009999999776482582]
'''