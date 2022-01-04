#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 16:45:53 2022

@author: ignacioalmodovarcardenas
"""


import pandas as pd
import numpy as np


train = pd.read_pickle('/Users/ignacioalmodovarcardenas/Desktop/Advanced programming/hyper_parameter_tuning/traintestdata_pickle/trainst1ns16.pkl')
test = pd.read_pickle('/Users/ignacioalmodovarcardenas/Desktop/Advanced programming/hyper_parameter_tuning/traintestdata_pickle/testst1ns16.pkl')


train_target=train.loc[:,"energy"]
test_target=test.loc[:,"energy"]

train_train=train.iloc[:,0:300]
test_test=test.iloc[:,0:300]

#Take 30 columns at random
index_column_NA=np.random.choice(300,30)

#Take the total number of data and calculate its 10%, this will be the number of missing values that we should impute
nNAs_train=int(index_column_NA.shape[0]*train_train.shape[0]*0.1)
nNAs_test=int(index_column_NA.shape[0]*test_test.shape[0]*0.1)

#Create a random index with for the rows with dimension the number of na neccesary
index_rNA_train=np.random.choice(int(train_train.shape[0]),nNAs_train)
index_rNA_test=np.random.choice(int(test_test.shape[0]),nNAs_test)

#Create a random index for the columns taking values only in the ones selected random at first with dimension the number of na neccesary
index_cNA_train=np.random.choice(index_column_NA,nNAs_train)
index_cNA_test=np.random.choice(index_column_NA,nNAs_test)

#impute na 
for n in range(index_cNA_train.shape[0]):
    train_train.iloc[index_rNA_train[n],index_cNA_train[n]]=np.nan

for n in range(index_cNA_test.shape[0]):
    test_test.iloc[index_rNA_test[n],index_cNA_test[n]]=np.nan