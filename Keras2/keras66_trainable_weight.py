
# trainable weight 의미

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([1, 2, 3, 4, 5])

y = np.array([1, 2, 3, 4, 5])

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
print('='*50)
print(model.weights)
print('='*50)
print(model.trainable_weights)
print('='*50)
print(len(model.weights))           # 각 레이어의 w, b의 세트 3개
print(len(model.trainable_weights)) # 가중치 갱신

model.trainable = False             # 가중치 갱신을 하지 않겠다는 뜻

print(len(model.weights))    
print(len(model.trainable_weights))

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=100)