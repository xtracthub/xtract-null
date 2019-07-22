import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import ShuffleSplit


def score_model(model, X_test, y_test):
    return model.score(X_test, y_test)


def print_performance(all_y_test, all_y_pred, avg_method='macro'):
    print("accuracy: {}\nprecision: {}\nrecall: {}".format(
        accuracy_score(all_y_test, all_y_pred),
        precision_score(all_y_test, all_y_pred, average=avg_method),
        recall_score(all_y_test, all_y_pred, average=avg_method)
    ))


def cross_validation(model, X, y, splits=1000, certainty_threshold=None):
    """Runs cross-validation on a test set to find accuracy of model and prints results.
        :param model: (sklearn.model) model to test
        :param X: (np.array) data matrix
        :param y: (np.array) true value column vector
        :param splits: (int) number of times to perform cross-validation
        :param certainty_threshold: (float | None) if the model has a decision function
        such as SVC's distance to separating hyperplane, this will print statistics for
        only those observations within the threshold
        :returns: (list, np.array) null values and column vector with index of null in list"""

    all_y_test = np.zeros((0, 1))
    all_y_pred = np.zeros((0, 1))
    all_y_decision = np.zeros((0, 1))
    i = 1
    for train_inds, test_inds in ShuffleSplit(n_splits=splits,
                                              test_size=0.9).split(X, y):
        # Split off the train and test set
        X_test, y_test = X[test_inds, :], y[test_inds]
        X_train, y_train = X[train_inds, :], y[train_inds]

        # Train the model
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        y_decision = np.asarray([max(row) for row in y_proba]).reshape(-1, 1)
        y_pred = np.asarray([model.classes_[row.argmax()] for row in y_proba]).reshape(-1, 1)

        all_y_test = np.concatenate((all_y_test, y_test))
        all_y_pred = np.concatenate((all_y_pred, y_pred))
        all_y_decision = np.concatenate((all_y_decision, y_decision))

    print_performance(all_y_test, all_y_pred, avg_method='macro')

    if certainty_threshold is not None:
        print("---------------------------------------")
        print("Within certainty threshold of {}".format(certainty_threshold))
        print("---------------------------------------")

        print("certainty min: {}, max: {}".format(min(all_y_decision)[0], max(all_y_decision)[0]))
        outside_threshold = [i for i in range(0, all_y_decision.shape[0])
                             if all_y_decision[i][0] < certainty_threshold]
        # print outside_threshold
        all_y_test = np.delete(all_y_test, outside_threshold)
        all_y_pred = np.delete(all_y_pred, outside_threshold)
        print("number of samples within threshold: {} out of {}".format(all_y_pred.shape[0],
                                                                        all_y_pred.shape[0] + len(outside_threshold)))

        if all_y_pred.shape[0] > 0:
            print_performance(all_y_test, all_y_pred, avg_method='macro')
