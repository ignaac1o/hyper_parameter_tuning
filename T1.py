
import pandas as pd

from numpy.random import randint

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline





train = pd.read_pickle('traintestdata_pickle/trainst1ns16.pkl')
test = pd.read_pickle('traintestdata_pickle/testst1ns16.pkl')

train_target=train.loc[:,"energy"]
test_target=test.loc[:,"energy"]






train_train=train.iloc[:3650,0:75]
train_validation=train.iloc[3650:,0:75]
test1=test.iloc[:,0:75]


train_target_train=train_target.iloc[:3650]
train_target_validation=train_target.iloc[3650:]



pipeKNN = make_pipeline(StandardScaler(), KNeighborsRegressor())
pipeKNN.fit(train_train,train_target_train)
knn_pred=pipeKNN.predict(train_validation)





modelotree=DecisionTreeRegressor()
modelotree.fit(train_train,train_target_train)
tree_pred=modelotree.predict(train_validation)


pipeSVR = make_pipeline(StandardScaler(), SVR())
pipeSVR.fit(train_train,train_target_train)
SVR_pred=pipeSVR.predict(train_validation)



metrics.mean_absolute_error(knn_pred,train_target_validation)

metrics.mean_absolute_error(tree_pred,train_target_validation)

metrics.mean_absolute_error(SVR_pred,train_target_validation)


##### Evaluando