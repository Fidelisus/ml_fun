from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from plots import *

def get_grid_search(clf, par):
	return GridSearchCV(
		estimator=clf,
		param_grid=par,
		scoring = 'accuracy',
		verbose = 0,
		cv = 10,
		n_jobs=-1,
	)




def knn_optimal_parameters(knn_parameters, pp_data_full):
	knn_classifier = KNeighborsClassifier(algorithm='auto')

	grid_search_knn = get_grid_search(knn_classifier, knn_parameters)
	knn = grid_search_knn.fit(pp_data_full[0], pp_data_full[1])

	knn_plots(knn)
	return knn.best_params_, knn.best_score_


def nb_optimal_parameters(nb_parameters, pp_data_full):
	nb_classifier = GaussianNB()

	grid_search_nb = get_grid_search(nb_classifier, nb_parameters)
	nb = grid_search_nb.fit(pp_data_full[0], pp_data_full[1])

	nb_plots(nb)
	return nb.best_params_, nb.best_score_


def tree_optimal_parameters(tree_parameters, pp_data_full):
	tree_classifier = DecisionTreeClassifier()

	grid_search_tree = get_grid_search(tree_classifier, tree_parameters)
	tree = grid_search_tree.fit(pp_data_full[0], pp_data_full[1])

	tree_plots(tree)
	return tree.best_params_, tree.best_score_