import pandas as pd
from sklearn.model_selection import train_test_split

def rd_breastCancer(dataset_location, lines, testing_ratio, random_state):
	pass

def rd_diabetes(dataset_location, lines, testing_ratio, random_state):
	data = pd.read_csv(dataset_location+"diabetes.csv", nrows=lines)
	data.insert(0, 'ID', range(len(data)))
	data_training, data_testing = train_test_split(data, test_size=testing_ratio, random_state=random_state)
	return data_training, data_testing, ([],[],[],[0,1,2,3,4,5,6,7])

def rd_purchase(dataset_location, lines, testing_ratio, random_state):
	pass

def rd_speeddating(dataset_location, lines, testing_ratio, random_state):
	pass


def readingData(readingFunction, dataset_location, lines, testing_ratio, random_state):
	return readingFunction(dataset_location, lines, testing_ratio, random_state)