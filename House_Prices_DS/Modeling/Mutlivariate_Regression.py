#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:17:39 2020

@author: sahanaasokan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns 
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy import stats

#Lets apply the multivariable regression model to the data.
# Lets look at our P-Values and our VIF Values.


# Data Pre-Processing/ What Data is missing? Outliers?

Test_data = pd.read_csv('test.csv') #testing set
Train_data= pd.read_csv('train.csv')

missing_count= (Train_data.isnull().sum()/(1460)).sort_values(ascending=False)
missing_data=pd.concat([missing_count],axis=1,keys=['Missing Count'])


new_data = Train_data[['Neighborhood','LotArea','Foundation','OverallCond'
,'YearBuilt','OverallQual','TotalBsmtSF','GrLivArea','Functional','SalePrice']]

cleaned_data=Train_data.drop(columns=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'],axis=1)


# Applying the Model to our updated data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#Dummy variables for categorical variables

regression_data=pd.get_dummies(new_data, columns=['Neighborhood','Foundation','OverallQual'
 ,'OverallCond','YearBuilt', 'Functional'])


# Defining x and y for our training/test set
y=regression_data['SalePrice']
x=regression_data.drop(columns='SalePrice', axis=0)

#Split the Data
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size= 0.20 )

regressor = LinearRegression()
regressor.fit(xtrain,ytrain) 

# Predicting Test Results
y_pred= regressor.predict(xtest)


#Measured Ordinary Least Squares from our mode.
model = sm.OLS(ytrain,xtrain).fit()
results=model
print(results.summary())



from statsmodels.stats.outliers_influence import variance_inflation_factor
numerical_data=Train_data._get_numeric_data()
numerical_data=numerical_data.dropna()


Numerical_data_array = numerical_data.to_numpy()



values=[variance_inflation_factor(Numerical_data_array, i) for i in range(Numerical_data_array.shape[1])]
print(values)
VIF_Table=[values]

#Lets look at the VIF values for the variables we picked...
VIF_Table = pd.DataFrame(data=VIF_Table)
VIF_Table = pd.DataFrame.transpose(VIF_Table)
VIF_Table.index=[numerical_data.columns]


