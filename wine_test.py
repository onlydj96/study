import pandas as pd
import numpy as np
from pandas.core.reshape.reshape import get_dummies
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping

path = "../_data/dacon/wine/"
train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")

x = train.drop(columns=['id', 'quality'], axis=1)
y = train['quality']
test_file = test_file.drop(columns=['id'], axis=1)

print(train.corr())
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(data=train.corr(), square=True, annot=True, cbar=True)
plt.show()

# y = pd.get_dummies(y)

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# le.fit(x['type'])
# x['type'] = le.transform(x['type'])

# le.fit(test_file['type'])
# test_file['type'] = le.transform(test_file['type'])

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_file = scaler.transform(test_file)

# # print(np.unique(y)) # [4, 5, 6, 7, 8]
# # print(x.shape, y.shape) # (3231, 12) (3231,)



# model = load_model("./_save/dacon_wine_0.5966.h5")

# loss = model.evaluate(x_test, y_test)
# print("loss : ", loss)

# ############################## 제출용 #####################################

# result = model.predict(test_file)
# result_recover = np.argmax(result, axis=1).reshape(-1, 1) +4
# submit_file['quality'] = result_recover
# # print(result_recover[:20])
# # print(result_recover[:30])
# # print(np.unique(result_recover))
# submit_file.to_csv(path + "winequality.csv", index=False) 
