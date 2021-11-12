from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np


def knn_plots(knn):
    test_scores = knn.cv_results_['mean_test_score']
    param = knn.cv_results_['param_n_neighbors']

    plt.plot(param.data[::4], test_scores[::4], label='manhattan, uniform weights')
    plt.plot(param.data[2::4], test_scores[2::4], label='euclidean, uniform weights')
    plt.legend(loc='best')
    plt.xlabel("n_neighbors")
    plt.ylabel("accuracy")
    plt.show()

    plt.plot(param.data[1::4], test_scores[::4], label='manhattan, weighted')
    plt.plot(param.data[3::4], test_scores[2::4], label='euclidean, weighted')
    plt.legend(loc='best')
    plt.xlabel("n_neighbors")
    plt.ylabel("accuracy")
    plt.show()

def knn_optimal_parameters(pp_data_full):
    knn_classifier = KNeighborsClassifier(algorithm='auto')
    knn_parameters = {
        'n_neighbors': range(1,41),
        'p': (1,2),
        'weights': ('uniform', 'distance'),
    }

    # with GridSearch
    grid_search_knn = GridSearchCV(
        estimator=knn_classifier,
        param_grid=knn_parameters,
        scoring = 'accuracy',
        cv = 10,
    )

    knn = grid_search_knn.fit(pp_data_full[0], pp_data_full[1])

    # uncomment to get more stats
    best_accuracy = knn.best_score_
    print("Best accuracy from GridSearchCV", best_accuracy)
    knn_plots(knn)

    return knn.best_params_

def clf_kNN(pp_data_training, pp_data_full, random_state):
    # uncomment to search for the optimal parameters
    # knn_best_params = knn_optimal_parameters(pp_data_full)

    # optimal parameters for breastCancer
    knn_best_params = {'n_neighbors': 4, 'p': 2, 'weights': 'distance'}
    knn_optimal = KNeighborsClassifier(**knn_best_params)

    knn_optimal.fit(pp_data_training[0], pp_data_training[1])
    return knn_optimal

def nb_plots(nb):
    test_scores = nb.cv_results_['mean_test_score']
    param = nb.cv_results_['param_var_smoothing'].data

    plt.plot(param, test_scores)
    plt.xlabel("var_smoothing")
    plt.ylabel("accuracy")
    plt.show()

def nb_optimal_parameters(pp_data_full):
    nb_classifier = GaussianNB()

    params_NB = {'var_smoothing': np.logspace(1,-10, num=100)}
    grid_search_nb = GridSearchCV(estimator=nb_classifier, 
                    param_grid=params_NB, 
                    cv=10,
                    # verbose=1,
                    scoring='accuracy')

    nb = grid_search_nb.fit(pp_data_full[0], pp_data_full[1])

    # uncomment to get more stats
    best_accuracy = nb.best_score_
    print("Best accuracy from GridSearchCV", best_accuracy)
    nb_plots(nb)

    return nb.best_params_

def clf_naiveBayes(pp_data_training, pp_data_full, random_state):
    # uncomment to search for the optimal parameters
    # nb_best_params = nb_optimal_parameters(pp_data_full)

    # optimal parameters for breastCancer
    nb_best_params = {'var_smoothing': 0.278}
    nb_optimal = GaussianNB(**nb_best_params)

    nb_optimal.fit(pp_data_training[0], pp_data_training[1])
    return nb_optimal

def clf_optimal_parameters(pp_data_full, random_state):
    clf_classifier = DecisionTreeClassifier()

    params_clf = {
        'criterion': ['gini','entropy'],
        'max_depth': range(3, 12),
        'splitter': ['best', 'random'],
        'min_samples_split': np.arange(0.001, 0.15, 0.01).tolist(),
        'random_state': [123],
        # 'ccp_alpha': np.arange(0, 1, 0.1).tolist(),
    }
    grid_search_clf = GridSearchCV(estimator=clf_classifier, 
                    param_grid=params_clf, 
                    cv=10,
                    # verbose=1,
                    scoring='accuracy')

    clf = grid_search_clf.fit(pp_data_full[0], pp_data_full[1])

    # uncomment to get more stats
    best_accuracy = clf.best_score_
    print("Best accuracy from GridSearchCV", best_accuracy)

    return clf.best_params_

def clf_decisionTree(pp_data_training, pp_data_full, random_state):
    # uncomment to search for the optimal parameters
    # clf_best_params = clf_optimal_parameters(pp_data_full, random_state)
    # print(clf_best_params)

    # optimal parameters for breastCancer
    clf_best_params = {'criterion': 'gini', 'max_depth': 9, 'min_samples_split': 0.071, 'random_state': 123, 'splitter': 'random'}
    clf_optimal = DecisionTreeClassifier(**clf_best_params)

    clf_optimal.fit(pp_data_training[0], pp_data_training[1])
    return clf_optimal

def classify(classifierFunction, random_state, pp_data_training, pp_data_full):
	return classifierFunction(pp_data_training, pp_data_full, random_state)