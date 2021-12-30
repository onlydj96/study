# 훈련 데이터 10만개로 증폭
# 완료 후 기존 모델과 비교
# save_dir, _temp 추가
# 증폭 데이터는 temp에 저장, 훈련 끝난 후 결과보고 삭제

from tensorflow.keras.datasets import mnist
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_datagen = ImageDataGenerator(  
    rescale=1./255, 
    # horizontal_flip = True,
    #vertical_flip = True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    #rotation_range= 5,
    zoom_range = 0.1,
    #shear_range = 0.7,
    fill_mode= 'nearest')


# 증폭 데이터 생성
augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size = augment_size)

x_augmented = x_train[randidx].copy()  #copy() 메모리 생성
y_augmented = y_train[randidx].copy()  

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)

# 스케일러
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)


# 증폭한 데이터에 ImageDataGenerator를 사용
xy_train = train_datagen.flow(x_augmented, y_augmented, batch_size=augment_size, shuffle= False, save_to_dir="../_temp") 
# temp_storage = train_datagen.flow(x_augmented, y_augmented, batch_size=30, shuffle=True, save_to_dir="../_temp") 

x = np.concatenate((x_train, xy_train[0][0]))  # (100000, 28, 28, 1)
y = np.concatenate((y_train, xy_train[0][1]))  # (100000,)

#2. 모델
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(28, 28, 1)))
model.add(MaxPool2D())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# #3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) 

start = time.time()
model.fit(x, y, epochs=10, validation_split=0.3, steps_per_epoch=1000) # (100000/32)
end = time.time() - start
print("걸린 시간 : ", round(end, 2))

#4. 예측 
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)