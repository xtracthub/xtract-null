import pandas as pd
from train_model import get_text_rows
import numpy as np

# NOT DONE YET
def clean_predict_data(df):
    """Removes textual rows and fills zeros.
        :param X: (np.array) data matrix
        :param y: (np.array) true value column vector
        :returns: (np.array) cleaned matrix ready for model"""

    to_remove = get_text_rows(df)
    df = np.delete(df, to_remove, axis=0)

    return df


def predict_single_file(filename, trained_classifier):
    data = clean_predict_data(pd.read_csv(filename))
    predictions = {}

    for idx, line in data.iterrows():
        predictions.update({'Line {}'.format(idx):
                                trained_classifier.predict(line)})

    return predictions

