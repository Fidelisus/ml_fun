from enum import Enum, auto
from readingData import *
from preProcessing import *
from classifier import *

class Topic():
	def __init__(self, solutionFile, readingFunction, preProcessingFunction):
		self.solutionFile = solutionFile
		self.readingFunction = readingFunction
		self.preProcessingFunction = preProcessingFunction

class Classifier():
	def __init__(self, preProcessingFunction, classifierFunction):
		self.preProcessingFunction = preProcessingFunction
		self.classifierFunction = classifierFunction

breastCancer = Topic("breast-cancer-diagnostic.shuf", rd_breastCancer, pp_breastCancer)
diabetes = Topic("diabetes", rd_diabetes, pp_diabetes)
purchase = Topic("purchase600-100cls-15k", rd_purchase, pp_purchase)
speeddating = Topic("speeddating", rd_speeddating, pp_speeddating)

decisionTree = Classifier(pp_decisionTree, clf_decisionTree)
kNN = Classifier(pp_kNN, clf_kNN)
naiveBayes = Classifier(pp_naiveBayes, clf_naiveBayes)

parameter = {
	"fixed": {
		True: {
			kNN: {
				breastCancer: {'n_neighbors': 4, 'p': 1, 'weights': 'uniform'},
				diabetes: {'n_neighbors': 30, 'p': 2, 'weights': 'distance'},
				purchase: None,
				speeddating: None
			},
			naiveBayes: {
				breastCancer: {'var_smoothing': 0.21544346900318845},
				diabetes: {'var_smoothing': 0.464158883361278},
				purchase: None,
				speeddating: None
			},
			decisionTree: {
				breastCancer: {'criterion': 'gini', 'max_depth': 9, 'min_samples_split': 0.071, 'random_state': 123, 'splitter': 'random'},
				diabetes: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 0.15099999999999997, 'random_state': 123, 'splitter': 'best'},
				purchase: None,
				speeddating: None
			},
		},
		False: {
				kNN: {
				breastCancer: {'n_neighbors': 4, 'p': 1, 'weights': 'uniform'},
				diabetes: {'n_neighbors': 36, 'p': 1, 'weights': 'distance'},
				purchase: None,
				speeddating: None
			},
			naiveBayes: {
				breastCancer: {'var_smoothing': 0.21544346900318845},
				diabetes: {'var_smoothing': 0.001291549665014884},
				purchase: None,
				speeddating: None
			},
			decisionTree: {
				breastCancer: {'criterion': 'gini', 'max_depth': 9, 'min_samples_split': 0.071, 'random_state': 123, 'splitter': 'random'},
				diabetes: {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 0.15099999999999997, 'random_state': 123, 'splitter': 'best'},
				purchase: None,
				speeddating: None
			}
		}
	},
	"optimal": {
		kNN: {
			'n_neighbors': range(1,41),
			'p': (1,2),
			'weights': ('uniform', 'distance'),
		},
		naiveBayes: {
			'var_smoothing': np.logspace(1,-10, num=100)
		},
		decisionTree: {
			'criterion': ['gini','entropy'],
			'max_depth': range(1, 21),
			'splitter': ['best', 'random'],
			'min_samples_split': np.arange(0.001, 0.3, 0.01).tolist(),
			'random_state': [123]
		}
	}
}