import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, concatenate
from tensorflow.keras.callbacks import EarlyStopping

def split_xy5(dataset, time_steps, y_column):
    x,y = list(), list()  
    
    for i in range(len(dataset)):
        x_end_number= i + time_steps    
        y_end_number = x_end_number + y_column-1
    
        
        if y_end_number > len(dataset):  
            break
        
        tmp_x = dataset[i:x_end_number, :] 
        tmp_y = dataset[x_end_number+1: y_end_number, 1] 
        x.append(tmp_x)
        y.append(tmp_y)   
    return np.array(x),np.array(y)

#1. 데이터
path = "../samsung/"

samsung = pd.read_csv("D:/_data/stock predict/삼성전자.csv", index_col=0, header = 0, thousands =',', encoding='cp949')
kiwoom = pd.read_csv("D:/_data/stock predict/키움증권.csv", index_col=0, header = 0, thousands =',', encoding='cp949')

print(samsung)

# samsung = samsung.values
samsung = samsung.iloc[:250,:].sort_values(['일자'],ascending=[True])
kiwoom = kiwoom.iloc[:250,:].sort_values(['일자'],ascending=[True])

ss = samsung.loc[::-1].reset_index(drop=True)
ki = kiwoom.loc[::-1].reset_index(drop=True)

# print(samsung,kiwoom)   # 2020/12/15부터 데이터를 쓰겠다.

s = samsung[['시가','종가']].values
k = kiwoom[['시가','종가']].values
# print(s,k)    #시가와 종가만 가져왔다.

x1, y1 = split_xy5(s,5,1)
x2, y2 = split_xy5(k,5,1)

print(x1.shape)  # (245, 5, 2)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1,x2,y1,y2, train_size=0.8, shuffle=True, random_state=66)

print(x1_train.shape, x2_train.shape)   # (196, 5, 2) (196, 5, 2)

#2. 모델구성

#2-1 모델1
input1 = Input(shape=(5,2))
dense1 = LSTM(16, activation='relu')(input1)
dense2 = Dense(8, activation='relu')(dense1)
dense3 = Dense(4, activation='relu')(dense2)
output1 = Dense(1, activation='relu')(dense3)

#2-1 모델2
input2 = Input(shape=(5,2))
dense11 = LSTM(16, activation='relu')(input1)
dense12 = Dense(8, activation='relu')(dense11)
dense13 = Dense(4, activation='relu')(dense12)
output2 = Dense(1, activation='relu')(dense13)

merge1 = Concatenate(axis=1)([output1, output2])

#2-3 output모델1
output21 = Dense(8, activation='relu')(merge1)
output22 = Dense(12, activation='relu')(output21)
output23 = Dense(4, activation='relu')(output22)
last_output1 = Dense(1, activation='relu')(output23)

#2-4 output모델2
output31 = Dense(8, activation='relu')(merge1)
output32 = Dense(12, activation='relu')(output31)
output33 = Dense(4, activation='relu')(output32)
last_output2 = Dense(1, activation='relu')(output33)

model = Model(inputs=[input1,input2], outputs=[last_output1,last_output2])
model.summary()

#3. 컴파일, 훈련

model.compile(loss='mae', optimizer='adam') 

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)

model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=500, batch_size=1 ,verbose=1, validation_split=0.2, callbacks=[es]) 

# model.save("./save/keras.exam1.h5")


# #4. 평가, 예측

results = model.evaluate([x1_test, x2_test], [y1_test, y2_test])
print('loss : ', results)

y1_pred, y2_pred = model.predict([x1_test, x2_test])

print("ss : ", y1_pred[-1])
print("kw : ", y2_pred[-1])

# print('삼성전자 종가 : ', ss)
# print('키움증권 종가 : ', kw)

# # r21 = r2_score(y1_test, ss)
# # r22 = r2_score(y2_test, kw)

# # # result_ss = model.predict(ss)
# # # print(result_ss[-5:-1])

# # print('r2_1스코어 : ', r21)
# # print('r2_2스코어 : ', r22)