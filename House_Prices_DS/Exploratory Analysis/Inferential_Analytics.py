#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:03:23 2020

@author: sahanaasokan
"""
import numpy as np
import pandas as pd

# Lets apply Probability
 
import warnings
warnings.filterwarnings('ignore')

df= pd.read_csv('train.csv')


# What is the Proability of picking a house in the ____ Neighborhood
# Lets check for Timner and Sawyer


counts_neighborhood=df['Neighborhood'].value_counts()

#Total Number of Neighborhoods = 1460
Total_Counts= df['Neighborhood'].shape[0]
Sawyer_Counts= df[df['Neighborhood']=='Sawyer'].shape[0]
Timber_Counts= df[df['Neighborhood']=='Timber'].shape[0]


probability_Sawyer = (Sawyer_Counts/Total_Counts)*100
probability_Timber = (Timber_Counts/Total_Counts)*100



# Lets try some condiitonal probability
# given that we have the probability of picking a house in
# Sawyer, we go a step further and AGAIN pick a house from the SAME neighborhood

Picking_from_SawyerTwice=((Sawyer_Counts/Total_Counts) * ((Sawyer_Counts-1)/(Total_Counts-1)))*100



# PMF is a probability distribution for discrete random variable
# Normal Distribution is a common PDF for continous

#Inference
# Lets sample and take the mean of random 400 Houses
# This would be our subset

Random_mean = np.random.choice(a= df['SalePrice'], size=400)
print(Random_mean.mean())

import scipy.stats as stats
import math

sample_size = 800
sample = np.random.choice(a= df['SalePrice'],size = sample_size)

# Lets find the mean of the sample
sample_mean = sample.mean()

#Find the critical value
z_critical = stats.norm.ppf(q = 0.95)  

# Lets find the Standard Deviation of the population
population_std = df['SalePrice'].std()  


# checking the margin of error
margin_of_error = z_critical * (population_std/math.sqrt(sample_size)) 

# defining our confidence interval
confidence_interval = (sample_mean - margin_of_error,
                       sample_mean + margin_of_error)  


# lets calculate the true mean
true_mean= (df['SalePrice'].mean())

#95% of our sample's confidence intervals will contain the true mean

# We can conduct different trials and see the true mean,
#and see if they 95 percent do contain the true mean.

# Lets start with the T-test 
#Null_h= The mean of the houses in StoneBr are not different from the mean of the houses in the other neighorhoods.
stats.ttest_1samp(a= df[df['Neighborhood'] == 'StoneBr']['SalePrice'],               
                 popmean= df['SalePrice'].mean())  


#Let us do some hypothesis testing
#For testing we use the T-test or Z-test

# In this case lets do the mean of houses in sawyer is the same as the other houses 
# We first need to pick a significance level 0.05

from statsmodels.stats.weightstats import ztest

z_statistic, p_value = ztest(x1 = df[df['Neighborhood'] == 'Sawyer']['SalePrice'],
                             value = df['SalePrice'].mean())
print(z_statistic,p_value)
# P is < 0.05
# We can reject our null hypothesis.

#In statistical hypothesis testing, a type I error is the rejection of a true null hypothesis, while a type II error is the non-rejection of a false null hypothesis

# Chi Squared test of Independence

# H-null LandContour is not independent from SalePrice
# First we need to calculate the frequencies.

x=df['LandContour']
y = pd.qcut(df['SalePrice'], 3, labels = ['High', 'Medium', 'Low'])
contingency_table=pd.crosstab(x,y)

chi2, pval, dof, expected = stats.chi2_contingency(contingency_table)

print("ChiSquare test statistic: ",chi2)
print("p-value: ",pval)
















