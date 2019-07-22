import numpy as np
import pickle as pkl
import argparse
from pylab import rcParams
import sys
import datetime
from train_model import split_data, read_train_data, clean_data, save_model, train_model
from test_model import score_model
from predict import predict_single_file

rcParams['figure.figsize'] = 10, 7
current_time = datetime.datetime.today().strftime('%Y-%m-%d')
np.set_printoptions(threshold=sys.maxsize)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", type=float, default=0.8,
                        help="test/train split ratio")
    parser.add_argument("--training_file", type=str,
                        help="file to train from",
                        default="null_training_data.csv")
    parser.add_argument("--model_name", default="{}.pkl".format(current_time),
                        help="name to save model as", type=str)
    parser.add_argument("--predict_file", type=str, default=None,
                        help="file to predict based on a trained classifier")
    parser.add_argument("--trained_classifier", type=str,
                        help="trained classifier to predict on",
                        default='default_model.pkl')
    args = parser.parse_args()

    if args.predict_file is not None:
        # with open(args.trained_classifier, 'rb') as classifier_file:
        #     trained_classifier = pkl.load(classifier_file)
        # print(predict_single_file(args.predict_file, trained_classifier))
        pass
    else:
        X, y = read_train_data(args.training_file)
        X, y = clean_data(X, y)
        X_train, X_test, y_train, y_test = split_data(X, y, args.split)
        model = train_model(X_train, y_train)
        print("Model accuracy: {}".format(score_model(model, X_test, y_test)))
        save_model(model, args.model_name)


if __name__ == "__main__":
    main()
