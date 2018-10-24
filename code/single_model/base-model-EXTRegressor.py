#title: training model using Extra Tree Regression 
#author: ntcong
#date 22/10/2018

#This code aims to training and testing model predict the price by using Extra Tree Regression algorithm
#Input: data/Xtest.csv, data/Xtrain.csv and data/ytrain.csv
#Output: code/EXTsubmission_[score]_[time].csv


import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
 
def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5
RMSE = make_scorer(mean_squared_error_, greater_is_better=False)    
    
def create_submission(prediction,score):
    now = datetime.datetime.now()
    sub_file = 'EXTsubmission_'+str(score)+'_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'.csv'
    print ('Creating submission: ', sub_file)
    pd.DataFrame({'Id': Xtest['Id'].values, 'SalePrice': prediction}).to_csv(sub_file, index=False) 

def model_extra_trees_regression(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain
    
    etr = ExtraTreesRegressor(n_jobs=1, random_state=0)
    param_grid = {'n_estimators': [100], 'max_features': [50,55,60]}
    # n_estimators : The number of trees in the forest
    # max_features : The number of features to consider when looking for the best split

    model = GridSearchCV(estimator=etr, param_grid=param_grid, n_jobs=1, cv=4, scoring=RMSE)
    # n_jobs : Number of jobs to run in parallel.
    # cv : Determines the cross-validation splitting strategy.

    model.fit(X_train, y_train)
    print('Extra trees regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_


# read data, build model and do prediction
Xtrain = pd.read_csv("../../input/Xtrain_samples.csv")
Xtest = pd.read_csv("../../input/Xtest_samples.csv") 
ytrain = pd.read_csv("../../input/ytrain_samples.csv") 

ytrain = ytrain.values[:,1]

test_predict,score = model_extra_trees_regression(Xtrain,Xtest,ytrain)

create_submission(np.exp(test_predict),score)



