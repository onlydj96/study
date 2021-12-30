from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

#1. 데이터
docs = ['너무 재밋어요', '참 최고예요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다.', 
        '한 번 더 보고 싶네요', '글쎄요', '별로에요', '생각보다 지루해요', '연기가 어색해요', 
        '재미없어요', '너무 재미없다', '참 재밋네요', '예람이가 잘 생기긴 했어요'
]

# 긍정 1, 부정 0
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)  

x = token.texts_to_sequences(docs)
# print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, maxlen=5)


word_size = len(token.word_index)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

model = Sequential()
model.add(Embedding(27, 10))  
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=1)

#4. 결과
acc = model.evaluate(pad_x, labels)[1]
print("acc : ", acc)


########################### 실습 ##########################

x_predict = ['나는 반장이 정말 재미없다 정말']
token.fit_on_texts(x_predict)
x_predict = token.texts_to_sequences(x_predict)

result = model.predict(x_predict)
print(result)

if result >= 0.5 :
    print("긍정")
else :
    print("부정")