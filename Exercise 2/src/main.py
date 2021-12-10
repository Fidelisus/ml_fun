import json
from algorithm import *
from constants import *
from preProcessing import load_insurance_data



def runTraining():
	insurance_X_train, insurance_y_train, insurance_X_test, insurance_y_test = load_insurance_data()
	
	learnings, bestLearning = simulatedAnnealing(LISTOFCLFS, insurance_X_train, insurance_y_train)

	bestAlg, bestMSE, bestParams, bestTimes = bestLearning.getBest()

	output = {
		"Best Algorithm": bestAlg,
		"Best MSE": bestMSE,
		"Best Parameter": bestParams,
		"Time of best solution:": bestTimes
	}
	return json.dumps(output, indent=4)


print(runTraining())