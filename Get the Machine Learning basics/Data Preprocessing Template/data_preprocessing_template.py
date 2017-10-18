#Data Preprocessing

'''Steps to be followed
Step 1: Importing the Libraries(numpy, pandas, matplotlib)
Step 2: Importing the Dataset(Data.csv, here)
Step 3: Fixing missing values
Step 4: Encoding categorical data(countries, purchased-yes/no), both Independent as well as the dependent variable
Step 5: Splitting the training data and test data
Step 6: Feature scaling(Normalisation/Standardisation) Note that the training data is to be fitted as well as transformed whereas the test data is just transformed. Also, feature scaling is done only for X(independent variable)
'''

'''Step1, Step2, Step5, Step6 are important'''

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
