

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,              
    horizontal_flip = True,        
    # vertical_flip= True,           
    width_shift_range = 0.1,       
    height_shift_range= 0.1,       
    # rotation_range= 5,
    zoom_range = 0.3,              
    # shear_range=0.7,
    fill_mode = 'nearest',
    validation_split=0.3          
    )                   # set validation split

train = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/',
    target_size=(150,150),
    batch_size=2,
    class_mode='binary',
    subset='training') # set as training data

test = train_datagen.flow_from_directory(
    '../_data/image/horse-or-human/', # same directory as training data
    target_size=(150,150),
    batch_size=2,
    class_mode='binary',
    subset='validation') # set as validation data

print(type(train)) # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(train[0][0].shape)

# np.save("./_save_npy/keras48_2_datasets_x.npy", arr = train_datagen[0][0])
# np.save("./_save_npy/keras48_2_datasets_y.npy", arr = train_datagen[0][1])

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(150, 150, 3)))
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
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
import time


es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto', restore_best_weights=True)

start = time.time()
hist = model.fit(train, epochs=100, steps_per_epoch=72,    # steps_per_epoch = 전체 데이터 수 / batch = 160 / 5 = 32
                    validation_data=test,
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
sample_image = sample_directory + "horse.jpg"

print("-- Evaluate --")
scores = model.evaluate(test, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

print("-- Predict --")
image_ = image.load_img(str(sample_image), target_size=(150, 150))
x = image.img_to_array(image_)
x = np.expand_dims(x, axis=0)
x /=255.
images = np.vstack([x])
classes = model.predict(images, batch_size=40)
# y_predict = np.argmax(classes)#NDIMS

print(classes)
test.reset()
print(test.class_indices)
# {'horses': 0, 'humans': 1}
if(classes[0][0]<=0.5):
    horses = 100 - classes[0][0]*100
    print(f"당신은 {round(horses,2)} % 확률로 horse 입니다")
elif(classes[0][0]>0.5):
    human = classes[0][0]*100
    print(f"당신은 {round(human,2)} % 확률로 human 입니다")
else:
    print("ERROR")
    
    
'''
acc: 80.00%
-- Predict --
[[0.00041024]]
{'horse': 0, 'human': 1}
당신은 99.96 % 확률로 horse 입니다
'''