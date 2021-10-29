from sklearn.tree import DecisionTreeClassifier

def decisionTree(pp_data_training, random_state):
	clf = DecisionTreeClassifier(random_state=random_state)
	clf.fit(pp_data_training.iloc[:,:-1], pp_data_training.iloc[:,-1])
	return clf

def kNN():
	pass


def classify(classifier, random_state, pp_data_training):
	if classifier == "decisionTree":
		classifier_function = decisionTree
	elif classifier == "kNN":
		classifier_function = kNN
	elif classifier == "3":
		classifier_function = rd_3

	return classifier_function(pp_data_training, random_state)