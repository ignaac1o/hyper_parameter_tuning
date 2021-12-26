import pandas as pd

from numpy.random import randint

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,PredefinedSplit
import numpy as np




train = pd.read_pickle('traintestdata_pickle/trainst1ns16.pkl')
test = pd.read_pickle('traintestdata_pickle/testst1ns16.pkl')

train_target=train.loc[:,"energy"]
test_target=test.loc[:,"energy"]





train_train=train.iloc[:3650,0:75]
train_validation=train.iloc[3650:,0:75]
test1=test.iloc[:,0:75]


train_target_train=train_target.iloc[:3650]
train_target_validation=train_target.iloc[3650:]



pipeKNN = make_pipeline(StandardScaler(), KNeighborsRegressor(p=1))
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




############################


train_cv_index=np.zeros(train.shape[0])

train_cv_index[:3650] = -1
train_cv_index = PredefinedSplit(train_cv_index)

#####


n_estimators = [int(x) for x in np.linspace(start = 1, stop = 20, num = 20)] # number of trees in the random forest
max_features = ['auto', 'sqrt'] # number of features in consideration at every split
max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
min_samples_split = [2, 6, 10] # minimum sample number to split a node
min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node
bootstrap = [True, False] # method used to sample data points

random_grid = {
    #'n_estimators': n_estimators,

'max_features': max_features,

'max_depth': max_depth,

'min_samples_split': min_samples_split,

'min_samples_leaf': min_samples_leaf}

rf_random = RandomizedSearchCV(estimator = modelotree,
    param_distributions = random_grid,
               n_iter = 100, cv = train_cv_index, verbose=2, random_state=35, n_jobs = -1,
               scoring="neg_mean_absolute_error")


rf_random.fit(train.iloc[:,0:75],train_target)
rf_random_pred=rf_random.predict(train_validation)


metrics.mean_absolute_error(tree_pred,train_target_validation)
metrics.mean_absolute_error(rf_random_pred,train_target_validation)


# USing SVMs
kernel=["linear","poly","rbf","sigmoid"]
degree=[1,2,3,4]
C=[x for x in np.linspace(start = 0.1, stop = 10, num = 10)]
shrinking=[True]

random_grid = {
   "kernel":kernel,
   "degree":degree,
   "C":C,
   "shrinking":shrinking}



scaler1=StandardScaler().fit(train_train,train_target_train)
train_train_st=scaler1.transform(train_train)
train_st=scaler1.transform(train.iloc[:,0:75])

rs_svr = RandomizedSearchCV(estimator = SVR,
    param_distributions = random_grid,
    n_iter = 100, 
    cv = train_cv_index,
    verbose=2, random_state=35, n_jobs = -1,scoring="neg_mean_absolute_error")

rs_svr.fit(train_st,train_target)

SVR_pred_2=rs_svr.predict(scaler1.transform(train_validation))


metrics.mean_absolute_error(SVR_pred,train_target_validation)
metrics.mean_absolute_error(SVR_pred_2,train_target_validation)

#Using KNN
n_neighbors=[3,4,5,6,7]
weight=["uniform","distance"]
algorithm=["ball_tree","kd_tree","brute"]
leaf_size=[10,20,30,40]
p=[1]


param_grid={
    "n_neighbors":n_neighbors,
    #"weight":weight,
    "algorithm":algorithm,
    "leaf_size":leaf_size,
    "p":p
    }


knn_estimator_cv=KNeighborsRegressor()




rs_knn = GridSearchCV(estimator =knn_estimator_cv, param_grid=param_grid,
                       cv = train_cv_index, verbose=2, 
                      n_jobs = -1,scoring="neg_mean_absolute_error")





rs_knn.fit(train_st,train_target)

Knn_pred_2=rs_knn.predict(scaler1.transform(train_validation))


metrics.mean_absolute_error(knn_pred,train_target_validation)
metrics.mean_absolute_error(Knn_pred_2,train_target_validation)


metrics.mean_absolute_error(rs_svr.predict(scaler1.transform(train_validation)),train_target_validation)