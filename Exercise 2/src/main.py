import pandas as pd
import json
from time import time
from algorithm import *
from constants import *
from preProcessing import preProcess
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


################################################################################
################################ CONFIGURATION #################################
################################################################################
dataset_location="../../Datasets/"
lines=None

topic = diabetes
cv = 3
################################################################################


def runTraining(topic:Topic, cv=10, random_seed=234):
	data = pd.read_csv(dataset_location + topic.filename, nrows=lines)

	pp_data = preProcess(topic.preProcessingFunction, data)
	
	learnings, bestLearning = simulatedAnnealing(pp_data, cv)

	bestAlg, bestRMSE, bestParams, bestTimes = bestLearning.getBest()

	output = {
		"Best Algorithm": bestAlg,
		"Best RMSE": bestRMSE,
		"Best Parameter": bestParams,
		"Time of best solution:": bestTimes
	}
	return json.dumps(output, indent=4)


print(runTraining(topic, cv))