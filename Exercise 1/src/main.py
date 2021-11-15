import pandas as pd
import json
from time import time
from constants import *
from readingData import readingData
from preProcessing import preProcess
from classifier import classify
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


################################################################################
################################ CONFIGURATION #################################
################################################################################
dataset_location="./Datasets/"
lines=None

topic = diabetes
classifier = decisionTree
parameter_type = "optimal" # "fixed" / "optimal"
scaling = True
################################################################################


def runTraining(topic:Topic, classifier:Classifier, parameter, scaling = True, use_pca = False, testing_ratio=0.2, random_state=234):
	data_training, data_testing, cols = readingData(topic.readingFunction, dataset_location, lines, testing_ratio, random_state)

	data_training, data_testing, data_full = preProcess(topic.preProcessingFunction, data_training, data_testing, scaling)
	
	# data_training, data_testing = preProcess(classifier.preProcessingFunction, data_training, data_testing, cols=cols)

	# pp_data_training.to_csv(dataset_location+topic.solutionFile+".pp_training.csv")
	# pp_data_testing.to_csv(dataset_location+topic.solutionFile+".pp_testing.csv")

	pp_data_training, pp_data_testing, pp_data_full = data_training, data_testing, data_full



	#Add missing cols when one hot encoding
	if topic==speeddating:
		missing_cols = set(pp_data_training[0].columns) - set(pp_data_testing[0].columns)
		for c in missing_cols:
			pp_data_testing[0][c] = 0
		pp_data_testing = tuple((pp_data_testing[0][pp_data_training[0].columns], pp_data_testing[1]))

	clf, time_train = classify(classifier.classifierFunction, parameter, pp_data_training, pp_data_full, use_pca = use_pca)
	if not use_pca:
		start_predict = time()
		predictions = clf.predict(pp_data_testing[0])
		time_predict = time()-start_predict
	else:
		pca = PCA()
		start_predict = time()
		predictions = clf.predict(pca.fit_transform(pp_data_testing[0]))
		time_predict = time()-start_predict



	solution = pd.DataFrame(list(zip(pp_data_testing[1], predictions)), columns=['ID','class']).sort_values(by=['ID'])
	solution.to_csv(dataset_location+topic.solutionFile+".sol.csv", index = False)

	print("Time for training:", time_train)
	print("Time for predicting:", time_predict)
	print(confusion_matrix(pp_data_testing[1], predictions))

	if topic == purchase:
		score = {
			"accuracy": accuracy_score(pp_data_testing[1], predictions),
			"precision": precision_score(pp_data_testing[1], predictions, average='macro'),
			"recall": recall_score(pp_data_testing[1], predictions, average='macro')
		}
	else :
		score = {
			"accuracy": accuracy_score(pp_data_testing[1], predictions),
			"precision": precision_score(pp_data_testing[1], predictions),
			"recall": recall_score(pp_data_testing[1], predictions)
		}
	return json.dumps(score, indent=4)


if parameter_type == "fixed":
	parameter = (parameter_type, parameter[parameter_type][scaling][classifier][topic])
elif parameter_type == "optimal":
	parameter = (parameter_type, parameter[parameter_type][classifier])
else:
	raise Exception("parameter_type must be 'fixed' or 'optimal'")
use_pca = False

print(runTraining(topic, classifier, parameter, scaling, use_pca))