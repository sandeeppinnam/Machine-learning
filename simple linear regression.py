# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:15:47 2023

@author: sandeep pinnam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\sandeep pinnam\Downloads\data science classes\june\7th\7th\SIMPLE LINEAR REGRESSION\Salary_Data.csv")

x = data.iloc[:,:-1].values
y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_train ,y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience(Training set'))
plt.xlabel('YearsOfExperience')
plt.ylabel('Salary')
plt.show()


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('YearsOfExperience')
plt.ylabel('Salary')
plt.show()

m = regressor.coef_ 
m

c = regressor.intercept_
c

y12 = 9312.5 * 12 + 26780
y12

bias = regressor.score(x_train, y_train)
bias

varience = regressor.score(x_test,y_test)
varience


