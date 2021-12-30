from tensorflow.keras.preprocessing.text import Tokenizer

text = "나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다."


token = Tokenizer()
token.fit_on_texts([text])  # 텍스트를 인덱싱하다
print(token.word_index)

x = token.texts_to_sequences([text])  # {'진짜': 1, '마구': 2, '나는': 3, '매우': 4, '맛있는': 5, '밥을': 6, '먹었다': 7}
print(x)  # [[3, 1, 4, 5, 6, 1, 2, 2, 7]]


from tensorflow.keras.utils import to_categorical   # 0부터 시작하기 때문에 비추
word_size = len(token.word_index)
print("word_size : ", word_size)

x = to_categorical(x)  # 원핫인코딩을 통해서 라벨링된 단어들이 숫자에 따라서 가치가 부여되지 않게 만든다.
print(x)
print(x.shape) 
# [[[0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1.]]]
# (1, 9, 8)

