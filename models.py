from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from utils import mean_error
from sklearn.metrics import make_scorer
import time
#import xgboost as xgb

RMSE = make_scorer(mean_error, greater_is_better=False)

def gradient_boosting_tree_model(Xtrain, Ytrain, Xtest):
	log = {}

	start_time = time.time()

	gbt = GradientBoostingRegressor(random_state=0)
	param_grid = {}

	model = GridSearchCV(estimator=gbt, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
	model.fit(Xtrain, Ytrain.values.ravel())

	#print('Best params: ', end='')
	#print(model.best_params_)
	log['best_params_'] = model.best_params_ 

	#print('Best CV score: ', end='')
	#print(-model.best_score_)
	log['best_score_'] = model.best_score_

	Ytest = model.predict(Xtest)

	log['duration'] = time.time() - start_time

	return Ytest, log

def random_forest_model(Xtrain, Ytrain, Xtest):
	log = {}

	start_time = time.time()

	rf = RandomForestRegressor(n_jobs=-1, random_state=0)
	param_grid = {'n_estimators': [500]}

	model = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
	model.fit(Xtrain, Ytrain.values.ravel())

	#print('Best params: ', end='')
	#print(model.best_params_)
	log['best_params_'] = model.best_params_ 

	#print('Best CV score: ', end='')
	#print(-model.best_score_)
	log['best_score_'] = model.best_score_

	Ytest = model.predict(Xtest)

	log['duration'] = time.time() - start_time

	return Ytest, log

def extra_trees_model(Xtrain, Ytrain, Xtest):
	log = {}

	start_time = time.time()

	et = ExtraTreesRegressor(n_jobs=1, random_state=0)
	param_grid = {'n_estimators': [100], 'max_features': [50,55,60]}

	model = GridSearchCV(estimator=et, param_grid=param_grid, n_jobs=1, cv=4, scoring=RMSE)
	model.fit(Xtrain, Ytrain.values.ravel())

	#print('Best params: ', end='')
	#print(model.best_params_)
	log['best_params_'] = model.best_params_ 

	#print('Best CV score: ', end='')
	#print(-model.best_score_)
	log['best_score_'] = model.best_score_

	Ytest = model.predict(Xtest)

	log['duration'] = time.time() - start_time

	return Ytest, log

def xgboost_model(Xtrain, Ytrain, Xtest):
	log = {}

	start_time = time.time()

	xgb = xgb.XGBRegressor(seed=0)
	param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.05],
        'max_depth': [7, 9, 11],
        'subsample': [0.8],
        'colsample_bytree': [0.75,0.8,0.85],
    }

	model = GridSearchCV(estimator=xgb, param_grid=param_grid, n_jobs=1, cv=4, scoring=RMSE)
	model.fit(Xtrain, Ytrain.values.ravel())

	#print('Best params: ', end='')
	#print(model.best_params_)
	log['best_params_'] = model.best_params_ 

	#print('Best CV score: ', end='')
	#print(-model.best_score_)
	log['best_score_'] = model.best_score_

	Ytest = model.predict(Xtest)

	log['duration'] = time.time() - start_time

	return Ytest, log

