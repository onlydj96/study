

# http://www.kaggle.com/c/dogs-vs-cats/data

from typing import Sequence
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping

train_datagen = ImageDataGenerator(
                rescale=1./255, 
                horizontal_flip=True, 
                vertical_flip=True, 
                width_shift_range=0.1,
                height_shift_range=0.1,
                # rotation_range=5,
                zoom_range=0.3,
                # shear_range=0.7,
                fill_mode='nearest')


test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    'D:/_data/Image/cat_dog/training_set',
    target_size = (50, 50),
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True)

xy_test = test_datagen.flow_from_directory(
    'D:/_data/Image/cat_dog/test_set',
    target_size= (50, 50),
    batch_size = 32,
    class_mode = 'categorical')

'''
flow_from_directory 메소드를 사용하면 폴더구조를 그대로 가져와서 ImageDataGenerator 객체의 실제 데이터를 채워준다. 
이 데이터를 불러올 때 앞서 정의한 파라미터로 전처리를 한다.
'''

print(type(xy_train)) # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(xy_train[0][0].shape)


#2. 모델
model = Sequential()

model.add(Conv2D(32, (2,2), input_shape=(50, 50, 3)))
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
model.add(Dense(2, activation='softmax'))

#3. 컴파일 
model.compile(loss = "categorical_crossentropy", optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
hist = model.fit(xy_train, epochs=100, steps_per_epoch=20, validation_data=xy_train, validation_steps=40, callbacks=[es]) 

#4. 예측
sample_directory = '../_data/image/testfile/'
sample_image = sample_directory + "dog.jpg"

print("-- Evaluate --")
scores = model.evaluate_generator(xy_test, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
image_ = image.load_img(str(sample_image), target_size=(50, 50))
x = image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /=255.
images = np.vstack([x])
classes = model.predict(images, batch_size=40)

print(classes)
xy_test.reset()
print(xy_test.class_indices)
# {'cats': 0, 'dogs': 1}
if(classes[0][0]<=0.5):
    cat = 100 - classes[0][0]*100
    print(f"당신은 {round(cat,2)} % 확률로 고양이 입니다")
elif(classes[0][0]>=0.5):
    dog = classes[0][0]*100
    print(f"당신은 {round(dog,2)} % 확률로 개 입니다")
else:
    print("ERROR")