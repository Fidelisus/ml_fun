import pandas as pd
import json
from constants import *
from readingData import readingData
from preProcessing import preProcess
from classifier import classify
from sklearn.metrics import accuracy_score, precision_score, recall_score

dataset_location="./Datasets/"
lines=None

def getError(prediction, truth):
	if prediction and truth:
		return ERROR.TP
	if prediction and not truth:
		return ERROR.FP
	if not prediction and truth:
		return ERROR.FN
	if not prediction and not truth:
		return ERROR.TN


def runTraining(topic:Topic=diabetes, classifier:Classifier=decisionTree, testing_ratio=0.2, random_state=None):
	data_training, data_testing, cols = readingData(topic.readingFunction, dataset_location, lines, testing_ratio, random_state)
	
	data_training, data_testing = preProcess(topic.preProcessingFunction, data_training, data_testing)
	data_training, data_testing = preProcess(classifier.preProcessingFunction, data_training, data_testing, cols=cols)
	pp_data_training = data_training.sort_values(by=['ID'])
	pp_data_testing = data_testing.sort_values(by=['ID'])

	pp_data_training.to_csv(dataset_location+topic.solutionFile+".pp_training.csv")
	pp_data_testing.to_csv(dataset_location+topic.solutionFile+".pp_testing.csv")

	clf = classify(classifier.classifierFunction, random_state, pp_data_training)

	predictions = clf.predict(pp_data_testing.iloc[:,1:-1])
	solution = pd.DataFrame(list(zip(pp_data_testing.iloc[:,0], predictions)), columns=['ID','class']).sort_values(by=['ID'])
	solution.to_csv(dataset_location+topic.solutionFile+".sol.csv", index = False)

	score = {
		"accuracy": accuracy_score(pp_data_testing.iloc[:,-1], predictions),
		"precision": precision_score(pp_data_testing.iloc[:,-1], predictions),
		"recall": recall_score(pp_data_testing.iloc[:,-1], predictions)
	}
	return json.dumps(score, indent=4)



print(runTraining())
#runTraining()