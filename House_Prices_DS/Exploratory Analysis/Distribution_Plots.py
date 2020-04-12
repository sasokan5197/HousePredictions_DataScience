#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:56:42 2020

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


#MultiVariate Analysis
# We have to tet for Normality, Homoscedasticity,Linearity, and Absence of correlated errors
# We can test for Normality by graphing a distribution graph

Test_data = pd.read_csv('test.csv') #testing set
Train_data= pd.read_csv('train.csv',index_col=0) 

feature='LotArea'
sns.distplot(Train_data[(feature)], hist=True)
plot.show()

feature1 = 'GrLivArea'
sns.distplot(Train_data[(feature1)])
plot.show()

feature2 = 'TotalBsmtSF'
sns.distplot(Train_data[(feature2)])
plot.show()

feature3 = 'SalePrice'
sns.distplot(Train_data[(feature3)])
plot.show()
























