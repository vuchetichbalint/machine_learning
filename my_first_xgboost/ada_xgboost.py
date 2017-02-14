#!/usr/bin/env python

"""
This script builds a model for ADA fall of 2016.
More info about the dataset: https://inclass.kaggle.com/c/alkalmazott-adatelemzes-nagyhazi-feladat
"""

# OS packages
from __future__ import print_function
import csv
# Third-party packages
from sklearn import ensemble
import pandas as pd
import numpy as np
# Custom imports

from sklearn.datasets import load_svmlight_files
from sklearn.metrics import accuracy_score

from xgboost.sklearn import XGBClassifier



def create_predictions(params):
	train_set, test_set = params
	non_input_features = ['ID', 'TARGET']

	model = ensemble.GradientBoostingRegressor(loss='ls', n_estimators=100, max_depth=7, subsample=0.8, min_samples_leaf=5, learning_rate=0.03, random_state=2016)
	model = model.fit(train_set.drop(non_input_features, axis=1), train_set['TARGET'])
	test_set['TARGET'] = model.predict(test_set.drop(non_input_features, axis=1))

	print(model.feature_importances_)
	return test_set

if __name__ == "__main__":
	# config paths, varables etc.
	input_dir = '/home/balint/workspace/dmlab/ada_2016osz/input/'

	# read data
	train_df = pd.read_csv(input_dir + 'train.csv', sep='|')
	test_df = pd.read_csv(input_dir + 'test.csv', sep='|')
	test_df['TARGET'] = 0
	numerical_features = ['EGESZ_SZOBAK', 'EMELET', 'FELSZOBAK', 'KERT_TERULET', 'KERULET', 'SZINTEK', 'TERULET', 'TETOTER', 'ID','TARGET']
	categorical_features = ['ALLAPOT', 'FUTES', 'KILATAS', 'KOMFORT', 'LIFT', 'PARKOLAS', 'VAROSRESZ']

	# preprocess data
	n_rows = len(train_df)
	data = pd.concat([train_df, test_df]).reset_index().copy()

	# get dummy varables for categorical features
	for f in categorical_features:
		data[f] = data[f].str.replace('-', '').str.replace(' ', '') 
		data[f] = data[f].fillna('kitudja')
		dummies_df = pd.get_dummies(data[f])
		del data[f]
		data = pd.concat([data, dummies_df], axis=1, join='inner')

	data = data.fillna(-9999)
	del data['index']
	data.to_csv('data.csv', index=False)
	train_df = data.iloc[:n_rows]
	test_df = data.iloc[n_rows:]
	
	# make prediction
	preds = create_predictions([train_df, test_df])

	#postprocess
	preds.loc[preds['TARGET'] < 0,['TARGET']] = 0
	preds.loc[preds['TARGET'] > 1,['TARGET']] = 0.99

	res = preds[['ID', 'TARGET']]

	res.to_csv('output.csv', index=False)
