# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:38:10 2023

@author: sandeep pinnam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
dataset = pd.read_csv(r"C:\Users\sandeep pinnam\Downloads\data science classes\june\6th- ML\Data.csv")

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values


from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
 
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#preprocessing the data

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting and Training and testing the data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)





