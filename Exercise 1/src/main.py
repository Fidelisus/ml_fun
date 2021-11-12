import pandas as pd
import json
from constants import *
from readingData import readingData
from preProcessing import preProcess
from classifier import classify
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

dataset_location="./Datasets/"
lines=None

def getError(prediction, truth):
	if prediction and truth:
		return ERROR.TP
	if prediction and not truth:
		return ERROR.FP
	if not prediction and truth:
		return ERROR.FN
	if not prediction and not truth:
		return ERROR.TN

def descriptive_plot(data):
    std_scaler = StandardScaler()
    # plot box plot of every variable describing mean
    plt.rcParams['figure.figsize'] = [20, 3]
    data.DataFrame(std_scaler.fit_transform(data.iloc[:, 2:12]), columns=data.iloc[:, 2:12].columns).boxplot()

def runTraining(topic:Topic=breastCancer, classifier:Classifier=decisionTree, testing_ratio=0.2, random_state=234):
    data_training, data_testing, cols = readingData(topic.readingFunction, dataset_location, lines, testing_ratio, random_state)

    data_training, data_testing, data_full = preProcess(topic.preProcessingFunction, data_training, data_testing)
    # I did not need it for now
    # data_training, data_testing = preProcess(classifier.preProcessingFunction, data_training, data_testing, cols=cols)

    # Why do we sort it? Imo, we should do it
    # pp_data_training = data_training.sort_values(by=['ID'])
    # pp_data_testing = data_testing.sort_values(by=['ID'])

    # pp_data_training.to_csv(dataset_location+topic.solutionFile+".pp_training.csv")
    # pp_data_testing.to_csv(dataset_location+topic.solutionFile+".pp_testing.csv")

    pp_data_training, pp_data_testing, pp_data_full = data_training, data_testing, data_full

    clf = classify(classifier.classifierFunction, random_state, pp_data_training, pp_data_full)

    print(clf.score(pp_data_testing[0], pp_data_testing[1]))

    # predictions = clf.predict(pp_data_testing.iloc[:,1:-1])
    predictions = clf.predict(pp_data_testing[0])
    solution = pd.DataFrame(list(zip(pp_data_testing[1], predictions)), columns=['ID','class']).sort_values(by=['ID'])
    solution.to_csv(dataset_location+topic.solutionFile+".sol.csv", index = False)

    print(confusion_matrix(pp_data_testing[1], predictions))

    score = {
        "accuracy": accuracy_score(pp_data_testing[1], predictions),
        "precision": precision_score(pp_data_testing[1], predictions),
        "recall": recall_score(pp_data_testing[1], predictions)
    }
    return json.dumps(score, indent=4)



print(runTraining())
#runTraining()