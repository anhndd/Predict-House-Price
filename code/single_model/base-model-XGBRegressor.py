#title: training model using Extreme Gradient Boosting
#author: ntcong
#date 22/10/2018

#This code aims to training and testing model predict the price by using Extreme Gradient Boosting algorithm
#Input: data/train.csv & data/test.csv
#Output: code/submission_[ time ].csv


import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
 
def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5
RMSE = make_scorer(mean_squared_error_, greater_is_better=False)    
    
def create_submission(prediction,score):
    now = datetime.datetime.now()
    sub_file = 'submission_'+str(score)+'_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'.csv'
    print ('Creating submission: ', sub_file)
    pd.DataFrame({'SalePrice': prediction}).to_csv(sub_file, index=False) #'Id': test['Id'].values, 

def model_xgb_regression(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain 
    
    xgbreg = xgb.XGBRegressor(seed=0)
    param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.05],
        'max_depth': [7, 9, 11],
        'subsample': [0.8],
        'colsample_bytree': [0.75,0.8,0.85],
    }
    model = GridSearchCV(estimator=xgbreg, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('eXtreme Gradient Boosting regression...')
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

#Xtrain,Xtest,ytrain = data_preprocess(train,test)

test_predict,score = model_xgb_regression(Xtrain,Xtest,ytrain)

create_submission(np.exp(test_predict),score)



