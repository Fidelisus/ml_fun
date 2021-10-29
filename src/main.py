import pandas as pd
from sklearn.model_selection import train_test_split
from readingData import readingData
from preProcessing import preProcessing
from classifier import classify

def runTraining(dataset_location="./datasets/", lines=None, testing_ratio=0.2, topic="diabetes", classifier="decisionTree", random_state=None):
	data_training, data_testing = readingData(topic, dataset_location, lines)
	pp_data_training = preProcessing(topic, data_training)
	pp_data_testing = preProcessing(topic, data_testing)

	pp_data_training = preProcessing(classifier, pp_data_training)
	pp_data_testing = preProcessing(classifier, pp_data_testing)

	pp_data_training.to_csv(dataset_location+"preprocessed_training.csv")
	pp_data_testing.to_csv(dataset_location+"preprocessed_testing.csv")

	clf = classify(classifier, random_state, pp_data_training)
	return clf.score(pp_data_testing.iloc[:,:-1], pp_data_testing.iloc[:,-1])


#print(runTraining())


s=0
for i in range(100):
	s += runTraining()

print(s/100)