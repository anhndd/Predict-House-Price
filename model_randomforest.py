'''
@title: training model using Random Forest Algorithm
@author: anhndd
@date 20/10/2018

- Input: Xtrain, Ytrain, Xtest from csv files
- Output: a csv file contains Ytest with 2 columns (separated by ',')
	+ Id of test records
	+ Predicted SalePrice
'''

import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

def predict_randomforest(Xtrain, Ytrain, Xtest):
	rfr = RandomForestRegressor(n_jobs=1, random_state=0)
	param_grid = {'n_estimators': [500], 'max_features': [10,15,20,25], 'max_depth':[3,5,7,9,11]} # another param
	# n_estimators : The number of trees in the forest
	# max_features: The number of features to consider when looking for the best split
	# max_depth   : The maximum depth of the tree

	RMSE = make_scorer(mean_error, greater_is_better=False)
	model = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
	# estimator  : This is assumed to implement the scikit-learn estimator interface
	# param_grid : dict or list of dictionaries
	# n_job      : Number of jobs to run in parallel
	# cv         : Determines the cross-validation splitting strategy
	# scoring    : A single string or a callable to evaluate the predictions on the test set

	model.fit(Xtrain, Ytrain.values.ravel())     # Trains the model for a given dataset

	print('Random forecast regression...')
	print('Best Params:')
	print(model.best_params_)
	# best_params_ : Parameter setting that gave the best results on the hold out data
	print('Best CV Score:')
	print(-model.best_score_)
	# best_score_ : Mean cross-validated score of the best_estimator

	Ypredict = model.predict(Xtest)		# Call predict on the estimator with the best found parameters
	return Ypredict

def mean_error(ground_truth, prediction):
	return mean_squared_error(ground_truth, prediction) ** 0.5

def create_summission(Xtest_id, prediction):
	time = datetime.datetime.now()
	file_name = 'submission_' + str(time.strftime('%Y-%m-%d-%H-%M')) + '.csv'
	pd.DataFrame({
		'Id': Xtest_id,
		'SalePrice': prediction
		}).to_csv(file_name, index=False)

if __name__ == '__main__':

	#Some sample
	print('Loading data from file...', end='')
	Xtrain = pd.read_csv('Xtrain_samples.csv')
	Ytrain = pd.read_csv('ytrain_samples.csv')
	Xtest = pd.read_csv('Xtest_samples.csv')
	print('Done')

	print('Dropping Id columns in data...', end='')
	Xtrain = Xtrain.drop(['Id'], axis=1)
	Ytrain = Ytrain.drop(Ytrain.columns[0], axis=1)
	Xtest_id = Xtest['Id'] #To save in sumission file
	Xtest = Xtest.drop(['Id'], axis=1)
	print('Done')

	#print('Xtrain', Xtrain)
	#print('ytrain', Ytrain.values.ravel())
	#print('Xtest', Xtest)

	print('Using Random Forest to predict...', end='')
	Ypredict = predict_randomforest(Xtrain, Ytrain, Xtest)
	#print(Ypredict)
	print('Done')

	print('Writing results to file...', end='')
	create_summission(Xtest_id.values.ravel(), Ypredict)
	print('Done')

	print('SUCCESSFUL!')