# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:25:45 2017

@author: DIAMONDCORP
"""
#Decision Tree Regression



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


#fitting regression model to dataset
#create your regressor here
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, Y)




#predicting a new result with the regression model
Y_pred = regressor.predict(6.5)



#visualising the regression results(for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'black')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('truth or bluff (decision tree regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()




















