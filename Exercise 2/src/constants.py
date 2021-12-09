from preProcessing import *

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR

class Topic():
	def __init__(self, filename, preProcessingFunction):
		self.filename = filename
		self.preProcessingFunction = preProcessingFunction

class Classifier():
	def __init__(self, classifier, params):
		self.classifier = classifier
		self.params = params

breastCancer = Topic("breast-cancer-diagnostic.shuf.lrn.csv", pp_breastCancer)
diabetes = Topic("diabetes.csv", pp_diabetes)
purchase = Topic("purchase600-100cls-15k.lrn.csv", pp_purchase)
speeddating = Topic("speeddating.csv", pp_speeddating)

randomForestRegressor = Classifier(RandomForestRegressor, {	'n_estimators': range(10, 200),
															'max_features': ['auto', 'sqrt', 'log2'],
															'max_depth': range(10, 110),
															'min_samples_split': range(2, 21),
															'min_samples_leaf': range(1, 21),
															'bootstrap': [True, False]})
linearSVR = Classifier(LinearSVR, {	'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
									'dual': [True, False],
									'tol': np.logspace(1e-5, 1e-1, 5),
									'C': np.logspace(1e-5, 1e-1, 5)+np.linspace(1, 25, 5),
									'epsilon': np.logspace(1e-5, 1e-1, 5)})
sdgRegressor = Classifier(SGDRegressor, {	'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
											'penalty': ['elasticnet'],
											'alpha': np.logspace(1e-2, 0, 3) ,
											'learning_rate': ['invscaling', 'constant'] ,
											'fit_intercept': [True, False],
											'l1_ratio': np.linspace(0.0, 1.0, 5),
											'eta0': np.logspace(1e-2, 1, 3),
											'power_t': np.logspace(0, 100, 100)})

listOfClfs = [randomForestRegressor, linearSVR, sdgRegressor]