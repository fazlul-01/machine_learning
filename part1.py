import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

# filling the missing values
from sklearn.preprocessing import Imputer
imputer = Imputer( missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])


#categorical data
#****************
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#Encoding the column 1 & 4
lb_x = LabelEncoder()
x[:,0] = lb_x.fit_transform(x[:,0])

lb_y = LabelEncoder()
y = lb_y.fit_transform(y)

#adding dummy variable to column 1

ohe = OneHotEncoder(categorical_features  = [0])
x=ohe.fit_transform(x).toarray()
