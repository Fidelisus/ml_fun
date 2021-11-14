import pandas as pd
import json
from constants import *
from readingData import readingData
from preProcessing import preProcess
from classifier import classify
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

dataset_location="./"
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

def runTraining(topic:Topic=breastCancer, classifier:Classifier=naiveBayes, testing_ratio=0.2, random_state=234):
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

    #Add missing cols when one hot encoding
    if topic==speeddating or topic==speeddating_no:
        missing_cols = set(pp_data_training[0].columns) - set(pp_data_testing[0].columns)
        for c in missing_cols:
            pp_data_testing[0][c] = 0
        pp_data_testing = tuple((pp_data_testing[0][pp_data_training[0].columns], pp_data_testing[1]))


    use_pca = False
    clf = classify(classifier.classifierFunction, random_state, pp_data_training, pp_data_full, use_pca = use_pca)

   
    if not use_pca:
        predictions = clf.predict(pp_data_testing[0])
    else:
        pca = PCA()
        predictions = clf.predict(pca.fit_transform(pp_data_testing[0]))

    # predictions = clf.predict(pp_data_testing.iloc[:,1:-1])
    solution = pd.DataFrame(list(zip(pp_data_testing[1], predictions)), columns=['ID','class']).sort_values(by=['ID'])
    solution.to_csv(dataset_location+topic.solutionFile+".sol.csv", index = False)

    print(confusion_matrix(pp_data_testing[1], predictions))

    if topic == purchase:
        score = {
            "accuracy": accuracy_score(pp_data_testing[1], predictions),
            "precision": precision_score(pp_data_testing[1], predictions, average='micro'),
            "recall": recall_score(pp_data_testing[1], predictions, average='micro')
        }
    else :
        score = {
            "accuracy": accuracy_score(pp_data_testing[1], predictions),
            "precision": precision_score(pp_data_testing[1], predictions),
            "recall": recall_score(pp_data_testing[1], predictions)
        }
    return json.dumps(score, indent=4)



print(runTraining(topic=purchase,classifier=decisionTree))
#runTraining()