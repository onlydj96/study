import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

# print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)  # x_train, x_test를 행으로 합친다는 뜻

x = x.reshape(70000, 28*28)

# 실습
# pca를 통해 0.95 이상인 n_components가 몇개?

pca = PCA(n_components=28*28)  # 칼럼이 28*28개의 벡터로 압축이됨
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)   
# print(sum(pca_EVR)) 

cumsum = np.cumsum(pca_EVR)  
# print(cumsum[0])



'''
cumsum : 누적합, 차례대로 더해준다.
'''
print(np.argmax(cumsum >= 0.95)+1)  # 154, 0.95가 되는 시작부분 
print(np.argmax(cumsum >= 0.99)+1)  # 331, 0.99가 되는 시작부분
print(np.argmax(cumsum >= 0.999)+1)  # 486, 0.999가 되는 시작부분
print(np.argmax(cumsum+1)) # 713, 1이 되는 시작부분