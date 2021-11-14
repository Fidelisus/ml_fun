import pandas as pd
import preProcessing
from sklearn.model_selection import train_test_split

def rd_breastCancer(dataset_location, lines, testing_ratio, random_state):
    df = pd.read_csv(dataset_location + "breast-cancer-diagnostic.shuf.lrn.csv")
    data_training, data_testing = train_test_split(df, test_size=0.2, random_state=random_state)
    return data_training, data_testing, None # None because everything is ratio data, so it is not needed

def rd_diabetes(dataset_location, lines, testing_ratio, random_state):
	data = pd.read_csv(dataset_location+"diabetes.csv", nrows=lines)
	data.insert(0, 'ID', range(len(data)))
	data_training, data_testing = train_test_split(data, test_size=testing_ratio, random_state=random_state)
	return data_training, data_testing, ([],[],[],[0,1,2,3,4,5,6,7])

def rd_purchase(dataset_location, lines, testing_ratio, random_state):
    df = pd.read_csv(dataset_location + "purchase600-100cls-15k.lrn.csv")
    data_training, data_testing = train_test_split(df, test_size=0.2, random_state=random_state)
    return data_training, data_testing, None # None because everything is ratio data, so it is not needed

def rd_speeddating(dataset_location, lines, testing_ratio, random_state):
    df = pd.read_csv(dataset_location + "speeddating.csv")
    #df = preProcessing.ohe_speed(df)
    data_training, data_testing = train_test_split(df, test_size=0.2, random_state=random_state)
    return data_training, data_testing, None # None because everything is ratio data, so it is not needed


def readingData(readingFunction, dataset_location, lines, testing_ratio, random_state):
	return readingFunction(dataset_location, lines, testing_ratio, random_state)