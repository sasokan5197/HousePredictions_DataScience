#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:02:16 2020

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
#Bivariate Analysis



Train_data= pd.read_csv('train.csv')
x=Train_data.to_numpy()

 
ScaledValues_Saleprice = StandardScaler().fit_transform(Train_data['SalePrice'][:,np.newaxis]);
#new axis simply transforms the array 1d-1d 2d-3d etc
low_range = ScaledValues_Saleprice[ScaledValues_Saleprice[:,0].argsort()][:10]
high_range= ScaledValues_Saleprice[ScaledValues_Saleprice[:,0].argsort()][-10:]


var='TotalBsmtSF'
Train_data.plot.scatter(x=var, y='SalePrice',xlim=(0,6500), ylim=(0,1000000));
plot.show()


var='GrLivArea'
Train_data.plot.scatter(x=var, y='SalePrice', ylim=(0,1000000));
plot.show()


#Lets find the values that corresponding to GrlivArea > 4000
Id_check=Train_data.sort_values(by = 'GrLivArea', ascending = False)[:5]
# 1299,524. Lets drop these variables.


# We can then change our plot to a normal distribution by using the log function
# From these Distributions we Can see that they are NOT NORMAL.
# We can fix this by converting into normal by using log.

Log_SalePrice  = np.log(Train_data['SalePrice'])
Log_GrLivArea = np.log(Train_data['GrLivArea'])
Log_LotArea = np.log(Train_data['LotArea'])


#A lot of houses dont have basements so there are a lot of 0s
#We can either replace the 0 values with the median of basement values
# Or we can apply the log fucntion to only the houses that have a value >0

#Log_TotalBsmtSF = np.log(Train_data['TotalBsmtSF'])

Train_data['Has a Basement'] = pd.Series(len(Train_data['TotalBsmtSF']), 
index=Train_data.index)
Train_data['Has a Basement']=0
Train_data['Has a Basement']=Train_data.loc[Train_data['TotalBsmtSF'] > 0,'Has a Basement']=1


Train_data.loc[Train_data['Has a Basement']==1,'TotalBsmtSF'] = np.log(Train_data['TotalBsmtSF'])

sns.distplot(Train_data[Train_data['TotalBsmtSF']>0]['TotalBsmtSF']);
plot.show()

sns.distplot(Log_SalePrice);
plot.show()

sns.distplot(Log_GrLivArea);
plot.show()

sns.distplot(Log_LotArea);
plot.show()







