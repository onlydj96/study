

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
    rotation_range=0.5,
    zoom_range=0.3,
    # shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.3
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3)

batch_num = 5
train_generator = train_datagen.flow_from_directory(
    '../_data/image/rps/',
    target_size=(50, 50),                         # size는 원하는 사이즈로 조정해 줌. 단, 너무 크기 차이가 나면 안좋을 수 있음
    batch_size=batch_num,
    class_mode='categorical',
    subset='training',
    shuffle=True,
)       # Found 1764 images belonging to 3 classes.

validation_generator = test_datagen.flow_from_directory(
    '../_data/image/rps/',
    target_size=(50, 50),
    batch_size=batch_num,
    class_mode='categorical',
    subset='validation'    
)       # Found 756 images belonging to 3 classes.


# print(type(train)) # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(train[0][1].shape)

# np.save("./_save_npy/keras48_3_datasets_x.npy", arr = train[0][0])
# np.save("./_save_npy/keras48_3_datasets_y.npy", arr = train[0][1])

#2. 모델 구성
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
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
import time

es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)

start = time.time()
hist = model.fit(train_generator, epochs=200, steps_per_epoch=70,    # steps_per_epoch = 전체 데이터 수 / batch = 160 / 5 = 32
                    validation_data=validation_generator,
                    validation_steps=30, callbacks=[es]
                    )
end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')


#4. 평가, 예측
from tensorflow.keras.preprocessing import image

sample_directory = '../_data/image/test_file/'
sample_image = sample_directory + "rock.jpg"


print("-- Evaluate --")
scores = model.evaluate_generator(validation_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
image_ = image.load_img(str(sample_image), target_size=(50, 50, 3))
x = image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /= 255.
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
y_predict = np.argmax(classes)  #NDIMS

print(classes)
validation_generator.reset()
print(validation_generator.class_indices)
# {'paper': 0, 'rock': 1, 'scissors': 2}
if(y_predict==0):
    paper = classes[0][0]*100
    print(f"이것은 {round(paper,2)} % 확률로 보 입니다")
elif(y_predict==1):
    rock = classes[0][1]*100
    print(f"이것은 {round(rock,2)} % 확률로 바위 입니다")
elif(y_predict==2):
    scissors = classes[0][2]*100
    print(f"이것은 {round(scissors,2)} % 확률로 가위 입니다")
else:
    print("ERROR")
    
'''
acc: 100.00%
-- Predict --
[[6.0461037e-02 9.3899745e-01 5.4158847e-04]]
{'paper': 0, 'rock': 1, 'scissors': 2}
이것은 93.9 % 확률로 바위 입니다
'''