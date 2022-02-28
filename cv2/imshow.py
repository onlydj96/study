
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import requests  # 외부에 있는 이미지를 가져올 때 사용
from io import BytesIO

url = 'https://cdn.pixabay.com/photo/2018/10/01/09/21/pets-3715733_960_720.jpg'

response = requests.get(url)
pic = Image.open(BytesIO(response.content))

pic_arr = np.asarray(pic)


fname = 'dogs.bmp'
pic_arr = cv2.imread(fname, cv2.IMREAD_COLOR)

cv2.imshow('DOG', pic_arr)
cv2.waitKey(0)             # 이미지가 응답없음이 뜰 때
cv2.destroyAllWindows() 