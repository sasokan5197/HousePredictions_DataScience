#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:36:43 2020

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
from sklearn.preprocessing import StandardScaler



data = pd.read_csv('train.csv',index_col=0) 
data = data.drop(['PoolQC','Fence','MiscFeature','Alley','Street'], axis=1)

# Initialize what we are using for X and Y.
df=data.loc[:,['MSSubClass','LotArea','TotalBsmtSF','1stFlrSF','2ndFlrSF','OverallQual','OverallCond','CentralAir','Heating','RoofStyle','Foundation',
'YearBuilt','MasVnrArea','GrLivArea','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','WoodDeckSF','OpenPorchSF',
'3SsnPorch','GarageArea','MiscVal','MoSold','BsmtFullBath','BsmtHalfBath','PoolArea','YrSold','Neighborhood','HouseStyle','ExterCond','ExterQual'
]]
x=df

y=data.loc[:,'SalePrice']


#Lets get dummy variables for our categorical variables.
x = pd.get_dummies(x, columns=['CentralAir','Heating','RoofStyle','Foundation','Neighborhood','HouseStyle','ExterCond','ExterQual'
])

# Lets impute missing values with the mean
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
x = imp.fit_transform(x)


# Lets initialize the variables. Normalize Y.

y=y.to_numpy()
y_normalized = np.log(y)


import xgboost as xgb
regressor = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y_normalized,test_size= 0.20 )

regressor.fit(xtrain, ytrain)
y_pred_xgb = regressor.predict(xtest)


from sklearn.metrics import mean_squared_log_error

print("XGBoost score on training set: ", np.sqrt(mean_squared_log_error(ytest, y_pred_xgb)))



