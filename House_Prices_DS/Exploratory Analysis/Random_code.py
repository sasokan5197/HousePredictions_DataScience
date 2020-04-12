#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:36:43 2020

@author: sahanaasokan
"""
'''




df_most_common_imputed =['MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'
'Electrical','KitchenQual','GarageType','GarageQual','GarageCond','SaleCondition','SaleType','PavedDrive'
,'GarageFinish''MoSold']


'MSSubClass','MSZoning',
'Street','LotShape','LandContour','Utilities','LotConfig','LandSlope',
'Neighborhood','Condition1','Condition2','BldgType' ,'HouseStyle','OverallQual',
'OverallCond','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','ExterQual','ExterCond', 
'Foundation' ,'Heating','HeatingQC','CentralAir','Functional'





categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numerical_transformer, numeric_cols),
        ('categorical', categorical_transformer, categorical_cols)
    ])









'''