#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:04:51 2020

@author: sahanaasokan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from scipy import stats
import statsmodels.api as sm
from scipy import stats


# Import the dataset
data = pd.read_csv('train.csv',index_col=0) 
data = data.drop(['PoolQC','Fence','MiscFeature','Alley','Street'], axis=1)

# Initialize what we are using for X and Y.
X=data.loc[:,['MSSubClass','LotArea','TotalBsmtSF','1stFlrSF','2ndFlrSF','OverallQual','OverallCond','CentralAir','Heating','RoofStyle','Foundation',
'YearBuilt','MasVnrArea','GrLivArea','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','WoodDeckSF','OpenPorchSF',
'3SsnPorch','GarageArea','MiscVal','MoSold','BsmtFullBath','BsmtHalfBath','PoolArea','YrSold','Neighborhood','HouseStyle','ExterCond','ExterQual'
]]

Y=data.loc[:,'SalePrice']

X.info()


# Encode Categorical Variables
X = pd.get_dummies(X, columns=['CentralAir','Heating','RoofStyle','Foundation','Neighborhood','HouseStyle','ExterCond','ExterQual'
])


# Get the list of the column names from X.
feature_list=list(X.columns)


# Count with the missing Values.
missing_count=(X.isnull().sum()/(1460)).sort_values(ascending=False)
missing_data=pd.concat([missing_count],axis=1,keys=['Missing Count'])


# Replace the na values with the mean.
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
X = imp.fit_transform(X)


# Change the Dataframe into an array
x_array=np.array(X)
y_array=np.array(Y)
y_array=np.log(Y)

# Split the dataset into training and testing sets.
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x_array,y_array,test_size= 0.20 )

# Import the Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(max_depth=30, random_state=0,n_estimators=1000)
regressor.fit(xtrain, ytrain)
y_pred=regressor.predict(xtest)

# Calculate the RMSE before we implement feature importance.
from sklearn.metrics import mean_squared_log_error
print("Root Mean Squared Logarithmic Score:", np.sqrt(mean_squared_log_error( ytest, y_pred)))


# Extracting Feature Importances

importances=list(regressor.feature_importances_)

feature_importances=[(feature,round(importances,2)) 
for feature, importances in zip(feature_list, importances)]

# Sort the feature_imprtances 

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
for feature, importances in feature_importances:
    print('Variable:', feature)
    print(' importances:', importances)



# We need to find the most Important features
Important_Features=[feature[0] for feature in feature_importances[:15]] 
Important_Indexes=[feature_list.index(feature) for feature in Important_Features]



#Run regression on the new variables and measure the new RMS score.
from sklearn.model_selection import train_test_split
xtrain_important,xtest_important,ytrain_important,ytest_important = train_test_split(x_array[:,Important_Indexes],y_array,test_size= 0.20 )

# Import the Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(max_depth=30, random_state=0,n_estimators=1000)
regressor.fit(xtrain_important, ytrain_important)
y_pred_important=regressor.predict(xtest_important)




#Check RMSE after new model.
print("Root Mean Squared Logarithmic Score:", np.sqrt(mean_squared_log_error(ytest_important, y_pred_important)))
#Multicolinearity, lets look at the correlation between the used variables.

