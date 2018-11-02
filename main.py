'''
@author: Nguyen Viet Sang
@date: 31/10/2018
'''
from models import gradient_boosting_tree_model
from models import random_forest_model
from models import extra_trees_model
from models import xgboost_model
import pandas as pd
import numpy as np
from utils import create_submission
from utils import write_log

if __name__ == '__main__':
	print('Loading data from file...', end='')
	Xtrain = pd.read_csv('input/Xtrain_samples.csv')
	Ytrain = pd.read_csv('input/Ytrain_samples.csv')
	Xtest = pd.read_csv('input/Xtest_samples.csv')
	print('Done')

	print('Dropping Id columns in data...', end='')
	Xtrain = Xtrain.drop(['Id'], axis=1)
	Ytrain = Ytrain.drop(Ytrain.columns[0], axis=1)
	Xtest_id = Xtest['Id'] #To save in sumission file
	Xtest = Xtest.drop(['Id'], axis=1)
	print('Done')

	# print('Xtrain', Xtrain)
	# print('ytrain', Ytrain.values.ravel())
	# print('Xtest', Xtest)

	log = {}
	print('Training with gradient_boosting_tree_model...', end='')
	Ytest, gbt_log = gradient_boosting_tree_model(Xtrain, Ytrain, Xtest)
	log['gbt_log'] = gbt_log
	create_submission(Xtest_id, Ytest, 'gradient_boosting_tree_model')
	print('Done')

	print('Training with random_forest_model...', end='')
	Ytest, rf_log = random_forest_model(Xtrain, Ytrain, Xtest)
	log['rf_log'] = rf_log
	create_submission(Xtest_id, Ytest, 'random_forest_model')
	print('Done')

	print('Training with extra_trees_model...', end='')
	Ytest, et_log = extra_trees_model(Xtrain, Ytrain, Xtest)
	log['et_log'] = et_log
	create_submission(Xtest_id, Ytest, 'extra_trees_model')
	print('Done')

	print('Training with xgboost_model...', end='')
	#Ytest, xgb_log = xgboost_model(Xtrain, Ytrain, Xtest)
	#log['xgb_log'] = xgb_log
	#create_submission(Xtest_id, Ytest, 'xgboost_model')
	print('Done')

	print('Writing log...', end='')
	write_log(log)
	print('Done')

	print('SUCCESSFUL')