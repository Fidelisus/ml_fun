import pandas as pd
from constants import *
from readingData import readingData
from preProcessing import preProcess
from classifier import classify

dataset_location="./Datasets/"
lines=None

def runTraining(topic:Topic=diabetes, classifier:Classifier=decisionTree, testing_ratio=0.2, random_state=None):
	data_training, data_testing, cols = readingData(topic.readingFunction, dataset_location, lines, testing_ratio, random_state)
	
	data_training, data_testing = preProcess(topic.preProcessingFunction, data_training, data_testing)
	pp_data_training, pp_data_testing = preProcess(classifier.preProcessingFunction, data_training, data_testing, cols=cols)

	pp_data_training.to_csv(dataset_location+"preprocessed_training.csv")
	pp_data_testing.to_csv(dataset_location+"preprocessed_testing.csv")

	clf = classify(classifier.classifierFunction, random_state, pp_data_training)
	return clf.score(pp_data_testing.iloc[:,:-1], pp_data_testing.iloc[:,-1])


print(runTraining())