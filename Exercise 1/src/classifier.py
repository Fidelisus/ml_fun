from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import numpy as np
from json import dumps

from optimalParameter import *


def clf_kNN(parameter, pp_data_training, pp_data_full, use_pca):
	if parameter[0] == "optimal":
		knn_best_params = knn_optimal_parameters(parameter[1], pp_data_full)
	else:
		knn_best_params = parameter[1]

	knn_optimal = KNeighborsClassifier(**knn_best_params)
	knn_optimal.fit(pp_data_training[0], pp_data_training[1])
	return knn_optimal


def nb_pca(pp_data_training, pp_data_full):
	pipe = Pipeline(steps=[
						('pca', PCA()),
						('estimator', GaussianNB()),
						])

	parameters = {'estimator__var_smoothing': np.logspace(1,-10, num=100)}
	nb = GridSearchCV(pipe, parameters, scoring='accuracy', cv=10, n_jobs=-1).fit(pp_data_full[0], pp_data_full[1])

	print("Best accuracy from GridSearchCV", nb.best_score_)

	best_params = nb.best_params_
	print(best_params)
	return {"var_smoothing": best_params["estimator__var_smoothing"]}


def clf_naiveBayes(parameter, pp_data_training, pp_data_full, use_pca):
	if use_pca:
		nb_best_params = nb_pca(pp_data_training, pp_data_full)
		pca = PCA()

		nb_optimal = GaussianNB(**nb_best_params)
		nb_optimal.fit(pca.fit_transform(pp_data_training[0]), pp_data_training[1])
	else:
		if parameter[0] == "optimal":
			nb_best_params = nb_optimal_parameters(parameter[1], pp_data_full)
		else:
			nb_best_params = parameter[1]
		
		nb_optimal = GaussianNB(**nb_best_params)
		nb_optimal.fit(pp_data_training[0], pp_data_training[1])
	return nb_optimal


def clf_decisionTree(parameter, pp_data_training, pp_data_full, use_pca):
	if parameter[0] == "optimal":
		tree_best_params = tree_optimal_parameters(parameter[1], pp_data_full)
	else:
		tree_best_params = parameter[1]

	tree_optimal = DecisionTreeClassifier(**tree_best_params)
	tree_optimal.fit(pp_data_training[0], pp_data_training[1])
	return tree_optimal



def classify(classifierFunction, parameter, pp_data_training, pp_data_full, use_pca=False):
	return classifierFunction(parameter, pp_data_training, pp_data_full, use_pca)