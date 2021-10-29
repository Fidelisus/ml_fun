import pandas as pd


def pp_beastCancer(data):
	pass

def pp_diabetes(data):
	
	
	pp_data = data


	return pp_data

def pp_speeddating(data):
	pass

def pp_purchase(data):
	pass



def pp_decisionTree(data):
	pass

def pp_kNN(data):
	pass

def pp_naiveBayes(data):
	pass


def preProcessing(topicORclassifier, data):
	if topicORclassifier == "breastCancer":
		preProcessing_function = pp_breastCancer
	elif topicORclassifier == "diabetes":
		preProcessing_function = pp_diabetes
	elif topicORclassifier == "speeddating":
		preProcessing_function = pp_speeddating
	elif topicORclassifier == "purchase":
		preProcessing_function = pp_purchase
	if topicORclassifier == "decisionTree":
		preProcessing_function = pp_decisionTree
	elif topicORclassifier == "kNN":
		preProcessing_function = pp_kNN
	elif topicORclassifier == "naiveBayes":
		preProcessing_function = pp_naiveBayes

	return preProcessing_function(data)