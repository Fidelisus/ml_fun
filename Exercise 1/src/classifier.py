from sklearn.tree import DecisionTreeClassifier

def clf_decisionTree(pp_data_training, random_state):
	clf = DecisionTreeClassifier(random_state=random_state)
	clf.fit(pp_data_training.iloc[:,1:-1], pp_data_training.iloc[:,-1])
	return clf

def clf_kNN():
	pass

def clf_naiveBayes():
	pass


def classify(classifierFunction, random_state, pp_data_training):
	return classifierFunction(pp_data_training, random_state)