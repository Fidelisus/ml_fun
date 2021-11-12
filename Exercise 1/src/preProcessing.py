import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def pp_breastCancer(data, cols):
    def preprocess(data, is_kaggle_test = False):
        std_scaler = StandardScaler()
        data = data.rename(columns={c:c.strip() for c in data.columns})
        columns_to_drop = ["ID"]
        if not is_kaggle_test:
            columns_to_drop.append("class")
        df_trimmed = data.drop(columns_to_drop, axis=1)
        pp_data = pd.DataFrame(std_scaler.fit_transform(df_trimmed), columns=df_trimmed.columns)
        return pp_data

    is_kaggle_test = False

    X = preprocess(data, is_kaggle_test)
    Y = data["class"]
    
    return X, Y

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
	return preProcessingFunction(data_training, cols), preProcessingFunction(data_testing, cols), preProcessingFunction(pd.concat([data_training, data_testing]), cols)
