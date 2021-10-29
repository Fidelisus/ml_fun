from readingData import *
from preProcessing import *
from classifier import *

class Topic():
	def __init__(self, readingFunction, preProcessingFunction):
		self.readingFunction = readingFunction
		self.preProcessingFunction = preProcessingFunction

class Classifier():
	def __init__(self, preProcessingFunction, classifierFunction):
		self.preProcessingFunction = preProcessingFunction
		self.classifierFunction = classifierFunction

breastCancer = Topic(rd_breastCancer, pp_breastCancer)
diabetes = Topic(rd_diabetes, pp_diabetes)
purchase = Topic(rd_purchase, pp_purchase)
speeddating = Topic(rd_speeddating, pp_speeddating)

decisionTree = Classifier(pp_decisionTree, clf_decisionTree)
kNN = Classifier(pp_kNN, clf_kNN)
naiveBayes = Classifier(pp_naiveBayes, clf_naiveBayes)