
import pandas as pd

from numpy.random import randint

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import metrics







train = pd.read_pickle('traintestdata_pickle/trainst1ns16.pkl')
test = pd.read_pickle('traintestdata_pickle/testst1ns16.pkl')

train_target=train.loc[:,"energy"]
test_target=test.loc[:,"energy"]


train1=train.iloc[:,0:75]
test1=test.iloc[:,0:75]

modeloknn=KNeighborsRegressor()
modeloknn.fit(train1,train_target)

## hhhh=modelo1.predict(test1)


modelotree=DecisionTreeRegressor()
modelotree.fit(train1,train_target)

## hhhh=modelotree.predict(test1)

modelosvm=SVR()
modelosvm.fit(train1,train_target)


##### Evaluando




