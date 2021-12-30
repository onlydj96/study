from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

#1. 데이터 전처리

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train))  # 평론의 최대길이 :  2494
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train))  # 평론의 평균길이 :  238.71364

# 전처리
x_train = pad_sequences(x_train, padding='pre', maxlen=200, truncating='pre')   # trucating='pre' : ??
print(x_train.shape)
x_test = pad_sequences(x_test, padding='pre', maxlen=200, truncating='pre')

print(x_train.shape, y_train.shape)  # (25000, 200) (25000, 2)
print(x_test.shape, y_test.shape)    # (25000, 200) (25000, 2)

'''
훈련용 뉴스기사 : 250000
테스트용 뉴스기사 : 25000
카테고리 : 2 (good or bad)
'''

#2. 모델 구성
model = Sequential()
model.add(Embedding(10000, 128, input_length=200))  # (num_word, (아무거나), maxlen)
model.add(LSTM(128))
model.add(Dense(82))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='auto', patience=20, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es])

#4. 결과, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

'''
loss :  [0.34648871421813965, 0.8512399792671204]
'''

# ####################################################################################################

# word_to_index = imdb.get_word_index()
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