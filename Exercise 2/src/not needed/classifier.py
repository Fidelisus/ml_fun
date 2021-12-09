from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from time import time
import numpy as np

from optimalParameter import *


def clf_kNN(parameter, pp_data_training, pp_data_full, use_pca):
	if parameter[0] == "optimal":
		knn_best_params, knn_best_score = knn_optimal_parameters(parameter[1], pp_data_full)
	else:
		knn_best_params, knn_best_score = parameter[1], None

	knn_optimal = KNeighborsClassifier(**knn_best_params)

	start_train = time()
	knn_optimal.fit(pp_data_training[0], pp_data_training[1])
	time_train = time()-start_train
	return knn_optimal, knn_best_params, knn_best_score, time_train


def nb_pca(pp_data_training, pp_data_full):
	pipe = Pipeline(steps=[
						('pca', PCA()),
						('estimator', GaussianNB()),
						])

	parameters = {'estimator__var_smoothing': np.logspace(1,-10, num=100)}
	nb = GridSearchCV(pipe, parameters, scoring='accuracy', cv=10, n_jobs=-1).fit(pp_data_full[0], pp_data_full[1])

	nb_best_params = {"var_smoothing": nb.best_params_["estimator__var_smoothing"]}
	return nb_best_params, nb.best_score_


def clf_naiveBayes(parameter, pp_data_training, pp_data_full, use_pca):
	if use_pca:
		nb_best_params, nb_best_score = nb_pca(pp_data_training, pp_data_full)
		pca = PCA()

		nb_optimal = GaussianNB(**nb_best_params)
		start_train = time()
		nb_optimal.fit(pca.fit_transform(pp_data_training[0]), pp_data_training[1])
		time_train = time()-start_train
	else:
		if parameter[0] == "optimal":
			nb_best_params, nb_best_score = nb_optimal_parameters(parameter[1], pp_data_full)
		else:
			nb_best_params, nb_best_score = parameter[1], None
		
		nb_optimal = GaussianNB(**nb_best_params)
		start_train = time()
		nb_optimal.fit(pp_data_training[0], pp_data_training[1])
		time_train = time()-start_train
	return nb_optimal, nb_best_params, nb_best_score, time_train


def clf_decisionTree(parameter, pp_data_training, pp_data_full, use_pca):
	if parameter[0] == "optimal":
		tree_best_params, tree_best_score = tree_optimal_parameters(parameter[1], pp_data_full)
	else:
		tree_best_params, tree_best_score = parameter[1], None

	tree_optimal = DecisionTreeClassifier(**tree_best_params)
	start_train = time()
	tree_optimal.fit(pp_data_training[0], pp_data_training[1])
	time_train = time()-start_train
	return tree_optimal, tree_best_params, tree_best_score, time_train



def classify(classifierFunction, parameter, pp_data_training, pp_data_full, use_pca=False):
	return classifierFunction(parameter, pp_data_training, pp_data_full, use_pca)