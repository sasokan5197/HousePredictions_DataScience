#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:47:23 2020
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




#Import the Dataset.

Test_data = pd.read_csv('test.csv') #testing set
Train_data= pd.read_csv('train.csv',index_col=0) #training set
print(Train_data.columns)


#lets drop the columns that have >15percent missing data.

cleaned_data=Train_data.drop(columns=['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'],axis=1)


#Lets take a look at the overall  sale price.
Train_data['SalePrice'].describe()

#Lets see the distribution of SalePrice
sns.distplot(Train_data['SalePrice'])


#Lets create a new table with variables we are testing
index= Train_data.index
columns =Train_data.columns

new_data = Train_data[['Neighborhood','LotArea','Foundation','OverallCond'
,'YearBuilt','OverallQual','TotalBsmtSF','GrLivArea','Functional','SalePrice']]


#TYou can use a correlation heat map to get an idea of what feature might be useful
correlations=Train_data.corr()

fig = plot.subplots(figsize=(10,10))
figure=sns.heatmap(correlations,vmax= 0.9,cmap='Blues',square=True)

#Lets create a correlation map to see the exact values between the variables and SalePrice
Numeric_Traindata = Train_data._get_numeric_data()
correlations_numeric= Numeric_Traindata.corr()

fig = plot.subplots(figsize=(10,10))
sns.heatmap(correlations_numeric,vmax=0.9,cmap='hot',square=True)
plot.show()


#From this we can see that multicolinearity is an issue.There
#Are high correlations between other variables ex GarageCars and GarageArea is very high


#ScatterPlot 1 Neighborhood - Categorical
feature='Neighborhood'
plot.subplots(figsize=(16, 8))
fig=sns.boxplot(x=feature, y='SalePrice',linewidth=2.5,data= new_data)
fig.axis(ymin=0, ymax=800000);


#ScatterPlot 2 LotArea
feature='LotArea'
new_data.plot.scatter(feature, 'SalePrice', color='r')


#ScatterPlot  Foundation
feature='Foundation'
plot.subplots(figsize=(16, 8))
fig=sns.boxplot(x=feature, y='SalePrice',linewidth=2.5,data= new_data)
fig.axis(ymin=0, ymax=800000);

#ScatterPlot  OverallQual
feature='OverallQual'
plot.subplots(figsize=(16, 8))
fig=sns.boxplot(x=feature, y='SalePrice',linewidth=2.5,data= new_data)
fig.axis(ymin=0, ymax=800000);


#ScatterPlot OverallCond
feature='OverallCond'
plot.subplots(figsize=(16, 8))
fig=sns.boxplot(x=feature, y='SalePrice',linewidth=2.5,data= new_data)
fig.axis(ymin=0, ymax=800000);

#ScatterPlot TotalBsmtSF
feature='TotalBsmtSF'
new_data.plot.scatter(feature, 'SalePrice', color='blue')

#ScatterPlot YearBuilt
feature='YearBuilt'
plot.subplots(figsize=(16, 8))
fig=sns.boxplot(x=feature, y='SalePrice',linewidth=2.5,data= new_data)
fig.axis(ymin=0, ymax=800000);

#ScatterPlot GrLivArea
feature='GrLivArea'
new_data.plot.scatter(feature, y='SalePrice', color='pink')

#ScatterPlot Functional
feature='Functional'
plot.subplots(figsize=(16, 8))
fig=sns.boxplot(x=feature, y='SalePrice',linewidth=2.5,data= new_data)
fig.axis(ymin=0, ymax=800000);

# So now we have go graphs to get an idea of linearity.





















