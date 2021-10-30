from enum import Enum, auto
from readingData import *
from preProcessing import *
from classifier import *

class ERROR(Enum):
	TP=auto()
	FP=auto()
	FN=auto()
	TN=auto()

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