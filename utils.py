from sklearn.metrics import mean_squared_error
import datetime
import pandas as pd
import json

def mean_error(ground_truth, prediction):
	return mean_squared_error(ground_truth, prediction) ** 0.5

def create_submission(Xtest_id, prediction, model_name):
	time = datetime.datetime.now()
	file_name = 'output/submission_' + model_name + str(time.strftime('%Y-%m-%d-%H-%M')) + '.csv'
	pd.DataFrame({
		'Id': Xtest_id,
		'SalePrice': prediction
		}).to_csv(file_name, index=False)

def write_log(log):
	time = datetime.datetime.now()
	file_name = 'log/results_' + str(time.strftime('%Y-%m-%d-%H-%M')) + '.txt'
	with open(file_name, 'w') as f:
		json.dump(log, f)