import os
import cv2

# path = 'D:/Set14/'  # 드라이브의 루트 디렉토리

# dir_list = os.listdir(path)

def prepare_images(path):
    for file in os.listdir(path):
        img = cv2.imread(path + file)
            
        # h, w, c = img.shape
        
        
        # new_height = h/factor
        # new_width = w/factor
        
        # img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    
        cv2.imwrite('D:/_data/{}'.format(file), img)
        
prepare_images('D:/')
    
    
