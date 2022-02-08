
x = 2
y = 10
w = 0.5
lr = 0.05
epochs = 300

for i in range(epochs):
    predict = x * w
    loss = (predict - y) **2
    
    # 가중치와 epoch 도 넣어서 아래 print를 수정
    print("Loss : ", round(loss, 3), "\tPredict : ", round(predict, 3))
    
    up_predict = x * (w + lr)
    up_loss = (y - up_predict) ** 2
    
    down_predict = x * (w - lr)
    down_loss = (y - down_predict) ** 2
    
    if(up_loss > down_loss):
        w = w - lr
    else:
        w = w + lr
    

