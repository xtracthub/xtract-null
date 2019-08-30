import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from math import isnan
import sys

np.set_printoptions(threshold=sys.maxsize)


def read_train_data(training_data):
    data = pd.read_csv(training_data)

    X = data.iloc[:, 3:-1].values  # Everything but last column
    y = data.iloc[:, -1:].values  # Last column

    return X, y


def is_number_or_none(field):
    """Determine if a string is a number or NaN by attempting to cast to it a
    float.

    Parameter:
    field (str): A field.

    Return:
    (bool): Whether field can be cast to a number.
    """
    if field is None:
        return True
    try:
        float(field)
        return True
    except ValueError:
        return False


def get_text_rows(matrix):
    """Get indices of all rows that have non-numerical aggregates.
        :param matrix: (np.array) matrix of data
        :returns: (list(int)) list of text indices to remove"""

    to_remove = []
    for i in range(0, len(matrix)):
        # If not a number?
        if not np.vectorize(is_number_or_none)(matrix[i]).all() \
                or np.vectorize(lambda x: str(x).lower() == "nan" or x is None)(matrix[i]).all():
            to_remove.append(i)

    return to_remove


def fill_zeros(matrix):
    """Fills all NaN and infinite entries with zeros.
        :param matrix: (np.array) matrix of data
        :returns: (np.array) matrix with zeros filled"""

    num_rows, num_cols = matrix.shape
    output_matrix = np.empty(matrix.shape)
    for i in range(0, num_rows):
        for j in range(0, num_cols):
            # If entry is not a number or really big
            if matrix[i][j] is None or isnan(float(matrix[i][j])) or float(matrix[i][j]) > np.finfo(np.float64).max:
                output_matrix[i][j] = np.float64(0)
            else:
                output_matrix[i][j] = matrix[i][j]

    return output_matrix


def clean_data(X, y):
    """Removes textual rows and fills zeros.
        :param X: (np.array) data matrix
        :param y: (np.array) true value column vector
        :returns: (np.array) cleaned matrix ready for model"""

    to_remove = get_text_rows(X)
    X = np.delete(X, to_remove, axis=0)
    y = np.delete(y, to_remove, axis=0)

    X = fill_zeros(X)
    y = fill_zeros(y)
    return X, np.ravel(y)


def split_data(X, y, split=0.8):
    # test_inds = int(len(X) * split)
    # X_test, y_test = X[test_inds, :], y[test_inds]
    # X_train, y_train = X[train_inds, :], y[train_inds]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    return X_train, X_test, y_train, y_test


def train_model(X, y):
    model = KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                 metric='euclidean',
                                 metric_params=None, n_jobs=1, n_neighbors=9,
                                 weights='distance')

    return model.fit(X, y)


def save_model(model, file_name):
    with open(file_name, "wb") as f:
        pkl.dump(model, f)
