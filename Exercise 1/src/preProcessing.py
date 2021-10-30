import pandas as pd


def pp_breastCancer(data, cols):
	pp_data = data
	return pp_data

def pp_diabetes(data, cols):
	data.iloc[:,-1] = data.iloc[:,-1].astype(bool)
	return data

def pp_purchase(data, cols):
	pp_data = data
	return pp_data

def pp_speeddating(data, cols):
	pp_data = data
	return pp_data



def pp_decisionTree(data, cols):
	pp_data = data
	return pp_data

def pp_kNN(data, cols):
	pp_data = data
	return pp_data

def pp_naiveBayes(data, cols):
	pp_data = data
	return pp_data


def preProcess(preProcessingFunction, data_training, data_testing, cols=None):
	return preProcessingFunction(data_training, cols), preProcessingFunction(data_testing, cols)