import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np


def knn_plots(knn):
	test_scores = knn.cv_results_['mean_test_score']
	param = knn.cv_results_['param_n_neighbors']

	plt.plot(param.data[::4], test_scores[::4], label='manhattan, uniform weights')
	plt.plot(param.data[2::4], test_scores[2::4], label='euclidean, uniform weights')
	plt.legend(loc='best')
	plt.xlabel("n_neighbors")
	plt.ylabel("accuracy")
	plt.show()

	plt.plot(param.data[1::4], test_scores[::4], label='manhattan, weighted')
	plt.plot(param.data[3::4], test_scores[2::4], label='euclidean, weighted')
	plt.legend(loc='best')
	plt.xlabel("n_neighbors")
	plt.ylabel("accuracy")
	plt.show()

def nb_plots(nb):
	test_scores = nb.cv_results_['mean_test_score']
	param = nb.cv_results_['param_var_smoothing'].data

	plt.plot(param, test_scores)
	plt.xlabel("var_smoothing")
	plt.ylabel("accuracy")
	plt.show()

def tree_plots(tree):
	tree_max_depth_plot(tree)
	tree_ccp_alpha_plot(tree)
	tree_min_samples_split_plot(tree, max_depths = [5, 7, 9])

def tree_ccp_alpha_plot(tree):
	params = pd.DataFrame(tree.cv_results_['params'])
	indices = params.index[(params['criterion']=='gini') &
							(params['max_depth']==9) &
							(params['splitter']=='random') &
							(params['min_samples_split']==0.071) &
							(params['random_state']==123)].tolist()
	param = params['ccp_alpha'].iloc[indices]
	test_scores = pd.DataFrame(tree.cv_results_['mean_test_score']).iloc[indices]

	plt.plot(param.values, test_scores.values)
	plt.xlabel("ccp_alpha")
	plt.ylabel("accuracy")
	plt.show()

def tree_max_depth_plot(tree):
	params = pd.DataFrame(tree.cv_results_['params'])
	indices = params.index[(params['criterion']=='gini') &
							(params['splitter']=='random') &
							(params['min_samples_split']==0.071) &
							(params['random_state']==123)].tolist()
	param = params['max_depth'].iloc[indices]
	test_scores = pd.DataFrame(tree.cv_results_['mean_test_score']).iloc[indices]

	plt.plot(param.values, test_scores.values, label = "min_samples_split = 0.071")

	plt.xlabel("max_depth")
	plt.ylabel("accuracy")
	plt.legend(loc='right')
	plt.show()

def tree_min_samples_split_plot(pp_data_full, max_depths = [5, 7, 9]):
	for depth in max_depths:
		params = pd.DataFrame(tree.cv_results_['params'])
		indices = params.index[(params['criterion']=='gini') &
								(params['max_depth']==depth) &
								(params['splitter']=='random') &
								(params['min_samples_split']==0.071) &
								(params['random_state']==123)].tolist()
		param = params['min_samples_split'].iloc[indices]
		test_scores = pd.DataFrame(tree.cv_results_['mean_test_score']).iloc[indices]

		plt.plot(param.values, test_scores.values, label = f"max_depth = {depth}")

	plt.xlabel("min_samples_split")
	plt.ylabel("accuracy")
	plt.legend(loc='right')
	plt.show()
