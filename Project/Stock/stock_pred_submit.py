import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

ss = pd.read_csv("D:/_data/stock predict/삼성전자.csv", thousands=',', encoding='CP949')
ss = ss.drop(range(20, 1120), axis=0)
ki = pd.read_csv("D:/_data/stock predict/키움증권.csv", thousands=',', encoding='CP949')
ki = ki.drop(range(20, 1060), axis=0)

# 인덱스 재배열
ss = ss.loc[::-1].reset_index(drop=True)
ki = ki.loc[::-1].reset_index(drop=True)


# 필요한 컬럼만 남겨두기
x_ss = ss.drop(['일자', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', 
                '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis =1)
x_ss = np.array(x_ss)
x_ki = ki.drop(['일자', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
                '신용비',  '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis =1)
x_ki = np.array(x_ki)


# split 함수 정의
def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column-1
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, 1:]
        tmp_y = dataset[x_end_number-1:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x_ssp, y_ssp = split_xy(x_ss, 3, 4)
x_kip, y_kip = split_xy(x_ki, 3, 4)


# 삼성 데이터 train_test_split 적용
x1_train, x1_test, y1_train, y1_test = train_test_split(x_ssp, y_ssp, train_size=0.8, random_state=66)

# 키움 데이터 train_test_split 적용
x2_train, x2_test, y2_train, y2_test = train_test_split(x_kip, y_kip, train_size=0.8, random_state=66)


#2. 모델
#삼성 input
input1 = Input(shape=(3, 3))
dense1_1 = LSTM(32, activation='relu')(input1)
dense1_2 = Dense(16, activation='relu')(dense1_1)
output1 = Dense(4, activation='relu')(dense1_2)

#키움 input
input2 = Input(shape=(3, 3))
dense2_1 = LSTM(32, activation='relu')(input2)
dense2_2 = Dense(16, activation='relu')(dense2_1)
output2 = Dense(4, activation='relu')(dense2_2)

#앙상블
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2])

#삼성 out
output1_1 = Dense(16, activation='relu')(merge1)
output1_2 = Dense(8)(output1_1)
ss_output = Dense(4)(output1_2)

#키움 out
output2_1 = Dense(16, activation='relu')(merge1)
output2_2 = Dense(8)(output2_1)
ku_output = Dense(4)(output2_2)

model = Model(inputs=[input1, input2], outputs=[ss_output, ku_output])


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
model.compile(loss='mae', optimizer='adam')
es = EarlyStopping(monitor='val_loss', mode='auto', patience=30, restore_best_weights=True)
hist = model.fit([x1_train,x2_train], [y1_train, y2_train], epochs=1000, batch_size=1,
                 validation_split=0.3, callbacks=[es])



model.save("D:/Study/_save/stock_wed_mp_3.h5")

#4. 평가
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test])
print('loss : ', loss)

x_samsung = x_ss[17:20,:3].reshape(1, 3, 3)
x_kiwoom = x_ki[17:20,:3].reshape(1, 3, 3)

ss_pred, ki_pred = model.predict([x_samsung, x_kiwoom])
samsung = ss_pred[-1][-1]
kiwoom = ki_pred[-1][-1]

print('예측값 : ', samsung, kiwoom)
