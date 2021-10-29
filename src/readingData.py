import pandas as pd

def rd_diabetes(dataset_location, lines):
	data = pd.read_csv(dataset_location+"diabetes.csv", nrows=lines)
	data_training, data_testing = train_test_split(data, test_size=testing_ratio, random_state=random_state)
	return data_training, data_testing

def rd_speeddating(dataset_location, lines):
	pass

def rd_3(dataset_location, lines):
	pass

def rd_4(dataset_location, lines):
	pass


def readingData(topic, dataset_location, lines):
	if topic == "3":
		readingData_function = rd_3
	elif topic == "diabetes":
		readingData_function = rd_diabetes
	elif topic == "speeddating":
		readingData_function = rd_speeddating
	elif topic == "4":
		readingData_function = rd_4

	return readingData_function(dataset_location, lines)