import os
import cv2
import matplotlib.pyplot as plt

path = 'D:/Set14/'

file_list = os.listdir(path)

print("file_list : {}".format(file_list))

img = cv2.imread('D:/Set14/baboon.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, c = img.shape

h = h/2
w = w/2

new_img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LINEAR)
cv2.imwrite('D:/', new_img)


# for file in os.listdir(path):
    
#     img = cv2.imread('D:/Set14/')
#     h, w, c = img.shape
    
#     height = h / 2
#     width = w / 2
    
#     new_img = cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR)
#     cv2.imwrite('D:/{}'.format(file), img)


