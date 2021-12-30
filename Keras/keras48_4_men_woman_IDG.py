# 과제
# 본인 사진으로 predict 하시오
# D:/_data 안에 넣고

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=0.6,
    zoom_range=0.4,
    # shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.3
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# D:\_data\image\brain

train_generator = train_datagen.flow_from_directory(
    '../_data/image/men_women/',
    target_size=(50, 50),                         # size는 원하는 사이즈로 조정해 줌. 단, 너무 크기 차이가 나면 안좋을 수 있음
    batch_size=10,
    class_mode='binary',
    subset='training',
    shuffle=True,
)       # Found 2317 images belonging to 2 classes.

validation_generator = train_datagen.flow_from_directory(
    '../_data/image/men_women/',
    target_size=(50,50),
    batch_size=10,
    class_mode='binary',
    subset='validation'    
)       # Found 992 images belonging to 2 classes.

print(train_generator[0][0].shape, train_generator[0][1].shape)   

# np.save("./_save_npy/keras48_4_datasets_x.npy", arr = datasets[0][0])
# np.save("./_save_npy/keras48_4_datasets_y.npy", arr = datasets[0][1])

#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(50, 50, 3)))
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
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss ='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
import time

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)

start = time.time()
hist = model.fit_generator(train_generator, epochs=100, steps_per_epoch=232,    # steps_per_epoch = 전체 데이터 수 / batch = 160 / 5 = 32
                    validation_data=validation_generator,
                    validation_steps=4, callbacks=[es]
                    )
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# 샘플 케이스 경로지정
#Found 1 images belonging to 1 classes.
sample_directory = '../_data/image/test_file/'
sample_image = sample_directory + "human.jpg"


print("-- Evaluate --")
scores = model.evaluate(validation_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
image_ = image.load_img(str(sample_image), target_size=(50, 50))
x = image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /=255.
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
# y_predict = np.argmax(classes)#NDIMS

print(classes)
validation_generator.reset()
print(validation_generator.class_indices)
# {'men': 0, 'women': 1}
if(classes[0][0]<=0.5):
    men = 100 - classes[0][0]*100
    print(f"당신은 {round(men,2)} % 확률로 남자 입니다")
elif(classes[0][0]>=0.5):
    women = classes[0][0]*100
    print(f"당신은 {round(women,2)} % 확률로 여자 입니다")
else:
    print("ERROR")

'''
acc: 68.00%
-- Predict --
[[0.64168745]]
{'men': 0, 'women': 1}
당신은 64.17 % 확률로 여자 입니다
'''