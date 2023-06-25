# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:20:33 2023

@author: sandeep pinnam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\sandeep pinnam\Downloads\data science classes\june\19th,20th\19th,20th\2.LOGISTIC REGRESSION CODE\logit classification.csv")

x = data.iloc[:,[2,3]].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.20,random_state=0) 


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtest = sc.fit_transform(xtest)
xtrain = sc.transform(xtrain)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(xtrain,ytrain)

ypred = classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(ytest, ypred)
print(ac)

from sklearn.metrics import classification_report
cr = classification_report(ytest,ypred)
cr

bias = classifier.score(xtrain,ytrain)
bias

varience = classifier.score(xtest,ytest)
varience


