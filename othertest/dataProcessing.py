import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
print(dataset)
# column accept the last one
x = dataset.iloc[:,:-1].values
# lastcolumn only
y = dataset.iloc[:,3].values
print(y)
# Filling missing values with mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean',verbose = 0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.fit_transform(x[:,1:3])

# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# lable_x = LabelEncoder()
# x[:,0]= lable_x.fit_transform(x[:,0])
# oneHotEncoder = OneHotEncoder(categories='auto')
# x = oneHotEncoder.fit_transform(x).toarray()
# print(x)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#categorising column
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.str)
print(x)

#transforming y with 0 and 1
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling



# from sklearn.preprocessing import StandardScaler
#
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

#
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train.reshape(-1,1))
