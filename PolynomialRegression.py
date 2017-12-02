# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:34:05 2017

@author: DIAMONDCORP
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#splitting dataset into training and test set

"""from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, 
                      random_state = 0)"""

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#fitting linear regression to the dataset (for comparison)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)


#fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)


#visualising the linear regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('truth or bluff (linear regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

#visualising the polynomial regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, Y, color = 'black')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'green')
plt.title('truth or bluff (polynomial regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()
  
#predicting a new result with linear regresssion

lin_reg.predict(6.5)

#predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))




























