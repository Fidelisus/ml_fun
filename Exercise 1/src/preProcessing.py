import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer

speedDatingNumeric = ["wave", "age", "age_o", "d_age", "importance_same_race", "importance_same_religion", "pref_o_attractive", "pref_o_sincere",
 "pref_o_intelligence", "pref_o_funny", "pref_o_ambitious", "attractive_o", "sinsere_o", "intelligence_o", "funny_o"
 , "ambitous_o", "shared_interests_o", "attractive_important", "sincere_important", "intellicence_important", "funny_important"
 , "pref_o_shared_interests","ambtition_important", "shared_interests_important", "attractive", "sincere", "intelligence", "funny", "ambition", "attractive_partner"
 , "sincere_partner", "intelligence_partner", "funny_partner", "ambition_partner", "shared_interests_partner", "sports",
  "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts"
  , "music", "shopping", "yoga", "interests_correlate", "expected_happy_with_sd_people", "expected_num_interested_in_me", "expected_num_matches", "like"
  , "guess_prob_liked", "met", "like", "like", "like"]


speedDatingNominal = ["gender", "d_d_age", "race", "race_o", "samerace", "d_importance_same_race", "d_importance_same_religion", "field", "d_pref_o_attractive"
, "d_pref_o_sincere", "d_pref_o_intelligence", "d_pref_o_funny", "d_pref_o_ambitious"
, "d_pref_o_shared_interests", "d_attractive_o", "d_sinsere_o", "d_intelligence_o", "d_funny_o", "d_ambitous_o",
 "d_shared_interests_o", "d_attractive_important", "d_sincere_important", "d_intellicence_important", "d_funny_important", "d_ambtition_important", "d_shared_interests_important"
, "d_attractive", "d_sincere", "d_intelligence", "d_funny", "d_ambition", "d_attractive_partner", "d_sincere_partner", "d_intelligence_partner", "d_funny_partner"
, "d_ambition_partner", "d_shared_interests_partner", "d_sports", "d_tvsports", "d_exercise", "d_dining", "d_museums", "d_art", "d_hiking"
, "d_gaming", "d_clubbing", "d_reading", "d_tv", "d_theater", "d_movies", "d_concerts", "d_music", "d_shopping", "d_yoga", "d_interests_correlate", "d_expected_happy_with_sd_people"
, "d_expected_num_interested_in_me", "d_expected_num_matches", "d_like", "d_guess_prob_liked"]


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
	def preprocess(data, is_kaggle_test = False):
		data = data.rename(columns={c:c.strip() for c in data.columns})
		columns_to_drop = ["ID"]
		if not is_kaggle_test:
			columns_to_drop.append("class")
		df_trimmed = data.drop(columns_to_drop, axis=1)
		return df_trimmed

	is_kaggle_test = False

	X = preprocess(data, is_kaggle_test)
	Y = data["class"]
	
	return X, Y

def pp_speeddating(data, cols):
	def preprocess(data, is_kaggle_test = False):
		std_scaler = StandardScaler()
		data = data.rename(columns={c:c.strip() for c in data.columns})
		columns_to_drop = ["decision", "decision_o", "has_null"]
		if not is_kaggle_test:
			columns_to_drop.append("class")
		df_trimmed = data.drop(columns_to_drop, axis=1)
		df_trimmed = df_trimmed.replace(to_replace="?", value=np.nan )
		imputerMean = SimpleImputer(strategy='mean', missing_values=np.nan)
		imputerFreq = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
		#compute missing values
		df_trimmed[speedDatingNumeric] = imputerMean.fit_transform(df_trimmed[speedDatingNumeric].values)
		df_trimmed[speedDatingNominal] = imputerFreq.fit_transform(df_trimmed[speedDatingNominal].values)
		#scale values
		df_trimmed[speedDatingNumeric] = std_scaler.fit_transform(df_trimmed[speedDatingNumeric].values)
		df_trimmed = ohe_speed(df_trimmed)

		return df_trimmed

	is_kaggle_test = True

	X = preprocess(data, is_kaggle_test)
	Y = data["match"]
	
	return X, Y

def pp_speeddating_noScale(data, cols):
	def preprocess(data, is_kaggle_test = False):
		std_scaler = StandardScaler()
		data = data.rename(columns={c:c.strip() for c in data.columns})
		columns_to_drop = ["decision", "decision_o", "has_null"]
		if not is_kaggle_test:
			columns_to_drop.append("class")
		df_trimmed = data.drop(columns_to_drop, axis=1)
		df_trimmed = df_trimmed.replace(to_replace="?", value=np.nan )
		imputerMean = SimpleImputer(strategy='median', missing_values=np.nan)
		imputerFreq = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
		#compute missing values
		df_trimmed[speedDatingNumeric] = imputerMean.fit_transform(df_trimmed[speedDatingNumeric].values)
		df_trimmed[speedDatingNominal] = imputerFreq.fit_transform(df_trimmed[speedDatingNominal].values)
		#scale values
		#df_trimmed[speedDatingNumeric] = std_scaler.fit_transform(df_trimmed[speedDatingNumeric].values)
		df_trimmed = ohe_speed(df_trimmed)

		return df_trimmed

	is_kaggle_test = True

	X = preprocess(data, is_kaggle_test)
	Y = data["match"]
	
	return X, Y
def ohe_speed(data):
	encoder = OneHotEncoder()
	data = data.rename(columns={c:c.strip() for c in data.columns})
	feature_arr = encoder.fit_transform(data[speedDatingNominal]).toarray()
	feature_labels = encoder.get_feature_names_out()
	feature_labels = np.array(feature_labels).ravel()
	df_trimmed = data.drop(speedDatingNominal, axis=1).reset_index(drop=True)
	df_enc = pd.DataFrame(feature_arr, columns=feature_labels)
	df_trimmed = pd.concat([df_trimmed,df_enc], axis=1)
	return df_trimmed

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
