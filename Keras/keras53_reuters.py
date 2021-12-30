from tensorflow.keras.datasets import reuters
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)


print(len(x_train), len(x_test))  # 8982 2246
print(y_train[0])  # 3
print(np.unique(y_train)) 
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

print(type(x_train), type(y_train))  # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_train.shape, y_train.shape)  # (8982,) (8982,)
print(len(x_train[0]), len(x_train[1]))  # 87 56

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train))  # 뉴스기사의 최대길이 :  2376
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train))  # 뉴스기사의 평균길이 :  145.5398574927633

# 전처리

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')   # trucating='pre' : ???>?
print(x_train.shape)
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')


# 원핫인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape)  # (8982, 100) (8982, 46)   
print(x_test.shape, y_test.shape)    # (2246, 100) (2246, 46)

'''
훈련용 뉴스기사 : 8982
테스트용 뉴스기사 : 2246
카테고리 : 46
'''

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten

model = Sequential()
model.add(Embedding(10000, 128, input_length=100))
model.add(LSTM(128))
model.add(Dense(82))
model.add(Dense(46, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es])

#4. 결과
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

'''
loss :  [1.6313310861587524, 0.6308993697166443]
'''
####################################################################################################

# word_to_index = reuters.get_word_index()
# # print(word_to_index)
# # print(sorted(word_to_index.items()))

# import operator
# print(sorted(word_to_index.items(), key = operator.itemgetter(1)))


# index_to_word = {}
# for key, value in word_to_index.item():
#     index_to_word[value+3] = key
    
# for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
#     index_to_word[index] = token
    
# print(' '.join([index_to_word[index] for index in x_train[0]]))