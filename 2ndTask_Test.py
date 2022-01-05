#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 16:45:53 2022

@author: ignacioalmodovarcardenas
"""


import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_regression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsRegressor

from sklearn import metrics

train = pd.read_pickle('/Users/ignacioalmodovarcardenas/Desktop/Advanced programming/hyper_parameter_tuning/traintestdata_pickle/trainst1ns16.pkl')
test = pd.read_pickle('/Users/ignacioalmodovarcardenas/Desktop/Advanced programming/hyper_parameter_tuning/traintestdata_pickle/testst1ns16.pkl')


train_target=train.loc[:,"energy"]
test_target=test.loc[:,"energy"]

train_train=train.iloc[:,0:300]
test_test=test.iloc[:,0:300]

#Split into train and train validation
train_split=train_train.iloc[:3650,:]
train_validation=train_train.iloc[3650:,:]

train_target_split=train_target.iloc[:3650]
train_target_validation=train_target.iloc[3650:]

train_NA_train=train_train.copy()
test_NA_test=test_test.copy()

#Take 30 columns at random
index_column_NA=np.random.choice(300,30)

#Take the total number of data and calculate its 10%, this will be the number of missing values that we should impute
nNAs_train=int(index_column_NA.shape[0]*train_NA_train.shape[0]*0.1)
nNAs_test=int(index_column_NA.shape[0]*test_NA_test.shape[0]*0.1)

#Create a random index with for the rows with dimension the number of na neccesary
index_rNA_train=np.random.choice(int(train_NA_train.shape[0]),nNAs_train)
index_rNA_test=np.random.choice(int(test_NA_test.shape[0]),nNAs_test)

#Create a random index for the columns taking values only in the ones selected random at first with dimension the number of na neccesary
index_cNA_train=np.random.choice(index_column_NA,nNAs_train)
index_cNA_test=np.random.choice(index_column_NA,nNAs_test)

#impute na 
for n in range(index_cNA_train.shape[0]):
    train_NA_train.iloc[index_rNA_train[n],index_cNA_train[n]]=np.nan

for n in range(index_cNA_test.shape[0]):
    test_NA_test.iloc[index_rNA_test[n],index_cNA_test[n]]=np.nan
    



###############################
#Deciede which imputation method is the best one

    
#Split into train and train validation
train_NA_split=train_NA_train.iloc[:3650,:]
train_NA_validation=train_NA_train.iloc[3650:,:]




imputerMean=SimpleImputer(strategy="mean")
imputerMedian=SimpleImputer(strategy="median")

scaler_MinMax=MinMaxScaler()
scaler_Robust=RobustScaler()

knn=KNeighborsRegressor()

clf=Pipeline([
    ("scale",scaler_MinMax),
    ("impute",imputerMean),
    ("knn",knn)
    ])

clf2=Pipeline([
    ("scale",scaler_Robust),
    ("impute",imputerMean),
    ("knn",knn)
    ])

clf3=Pipeline([
    ("scale",scaler_MinMax),
    ("impute",imputerMedian),
    ("knn",knn)
    ])

clf4=Pipeline([
    ("scale",scaler_Robust),
    ("impute",imputerMedian),
    ("knn",knn)
    ])

#1st Pipe Scale MinMax, impute mean  **BEST ONE**
clf.fit(train_NA_split,train_target_split)
train_validation_imp1=clf.predict(train_NA_validation)
metrics.mean_absolute_error(train_validation_imp1,train_target_validation)

#2nd Pipe Scale Robust, impute mean
clf2.fit(train_NA_split,train_target_split)
train_validation_imp2=clf2.predict(train_NA_validation)
metrics.mean_absolute_error(train_validation_imp2,train_target_validation)

#3rd Pipe Scale MinMax, impute median
clf3.fit(train_NA_split,train_target_split)
train_validation_imp3=clf3.predict(train_NA_validation)
metrics.mean_absolute_error(train_validation_imp3,train_target_validation)

#4th Pipe Scale Robust, impute median
clf4.fit(train_NA_split,train_target_split)
train_validation_imp4=clf4.predict(train_NA_validation)
metrics.mean_absolute_error(train_validation_imp4,train_target_validation)



###############################
#Hyp tunning for knn
selection=SelectKBest(f_regression)

pipe1=Pipeline([
    ("scale",scaler_MinMax),
    ("impute",imputerMean),
    ("select",selection),
    ("knn",knn)
    ])

param_grid={
    "select__k":[int(x) for x in np.linspace(start = 1, stop = 300, num = 50)],
    "knn__n_neighbors":[2,4,8,16,32]
    }

train_cv_index=np.zeros(train_NA_train.shape[0]) 
train_cv_index[:3650] = -1
train_cv_index = PredefinedSplit(train_cv_index)

grid_tunning=GridSearchCV(pipe1,param_grid,
                          scoring="neg_mean_squared_error",cv=train_cv_index,n_jobs=-1,verbose=1)

grid_tunning.fit(train_NA_train,train_target)
bestparams=grid_tunning.best_params_

pipe1.set_params(**bestparams)
pipe1.fit(train_NA_train,train_target)
a=pipe1["select"]
columns_bool=a.get_support()
a.scores_

train_NA_train.columns[columns_bool].shape

col_importance=pd.DataFrame({"Columns":train_NA_train.columns[columns_bool],
                             "Scores":a.scores_[columns_bool]})

col_importance.sort_values(by="Scores",ascending=False)



#PCA
pca=PCA()

pipe2=Pipeline([
    ("scale",scaler_MinMax),
    ("impute",imputerMean),
    ("pca",pca),
    ("knn",knn)
    ])

param_grid2={
    "pca__n_components":[int(x) for x in np.linspace(start = 1, stop = 300, num = 50)],
    "knn__n_neighbors":[2,4,8,16,32]
    }


grid_tunning2=GridSearchCV(pipe2,param_grid2,
                          scoring="neg_mean_squared_error",cv=train_cv_index,n_jobs=-1,verbose=1)

grid_tunning2.fit(train_NA_train,train_target)
grid_tunning2.best_params_


#### Evaluate model on the test

model1=grid_tunning.predict(test_NA_test)
model2=grid_tunning2.predict(test_NA_test)

metrics.mean_absolute_error(model1,test_target)
metrics.mean_absolute_error(model2,test_target)

#Aunque se perdio informacion se hizo mejor busqueda con el numero de 
#vecinos porque se usaron 16 en este caso como punto mas optimo contra 7 
#que se dio en la primera parte del trabajo











