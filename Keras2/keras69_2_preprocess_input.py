from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np



model = ResNet50(weights='imagenet')

img_path = 'D:/_data/cat-dog/*.jpg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)


# x라는 이미지에 차원을 추가한다.
x = np.expand_dims(x, axis=0)  # axis= 차원을 추가하려는 인덱스자리

# 전이학습 모델의 최적의 스케일러
x = preprocess_input(x)

# predict 값은 'imagenet'의 1000개의 이미지 분류 라벨을 확률적으로 나타낸다.
preds = model.predict(x)

print("결과 : ", decode_predictions(preds, top=5)[0])