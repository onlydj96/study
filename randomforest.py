import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
path = "../_data/dacon/wine/"
train = pd.read_csv(path + "train.csv")
test_file = pd.read_csv(path + "test.csv")
submit_file = pd.read_csv(path + "sample_submission.csv")

x = train.drop(columns=['id', 'quality'], axis=1)  #'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'free sulfur dioxide','total sulfur dioxide', 'pH', 'sulphates'
y = train['quality']
test_file = test_file.drop(columns=['id'], axis=1)
print(x.corr())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(x['type'])
x['type'] = le.transform(x['type'])

le.fit(test_file['type'])
test_file['type'] = le.transform(test_file['type'])

# y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

from sklearn.preprocessing import RobustScaler, MinMaxScaler

scaler = RobustScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

clf = RandomForestClassifier(n_estimators=300, max_depth=100,random_state=0)

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

clf.fit(x_train, y_train)

predict = clf.predict(x_test)
print(accuracy_score(y_test , predict))
