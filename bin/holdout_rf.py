import sys
sys.path.extend([".", ".."])
import argparse
import csv
import gzip
from cratenet.models import create_rf_model
from cratenet.dataset import load_gzipped_dataset
from sklearn.model_selection import train_test_split

try:
    import cPickle as pickle
except ImportError:
    import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="?", required=True, type=str,
                        help="the dataset to use (must be a .csv.gz file)")
    parser.add_argument("--predictions", nargs="?", required=True, type=str,
                        help="path to the output file containing the predictions for the holdout set; "
                             "a .csv extension should be used")
    parser.add_argument("--actual", nargs="?", required=True, type=str,
                        help="path to the output file containing the true values of the holdout set; "
                             "a .csv extension should be used")
    parser.add_argument("--holdout", nargs="?", required=False, type=float, default=0.10,
                        help="a number between 0. and 1. representing the proportion of the data that will form the "
                             "holdout set (optional, default is 0.10)")
    parser.add_argument("--seed", nargs="?", required=False, type=int, default=20012022,
                        help="the random seed to use (optional)")
    parser.add_argument("--model", nargs="?", required=False, type=str,
                        help="path to the persisted model; a .pkl.gz extension should be used (optional)")
    args = parser.parse_args()

    dataset_file = args.dataset
    predictions_file = args.predictions
    actual_file = args.actual
    holdout_percentage = args.holdout
    random_state = args.seed
    model_file = args.model

    print(f"performing a 90-10 holdout evaluation...")
    print(" ".join(f"model: {create_rf_model()}".replace("\n", "").split()))
    print(f"random seed: {random_state}")

    metadata, X, y = load_gzipped_dataset(dataset_file)
    print(f"num examples: {len(X):,}")

    regressor = create_rf_model()

    X_train, X_test, y_train, y_test, metadata_train, metadata_test = \
        train_test_split(X, y, metadata, random_state=random_state, shuffle=True,
                         test_size=holdout_percentage)

    print(f"training model on {len(X_train):,} examples...")
    regressor.fit(X_train, y_train)

    print(f"making predictions on {len(X_test):,} examples...")
    predicted_y = regressor.predict(X_test)

    print(f"writing predictions to {predictions_file}...")
    with open(predictions_file, "wt") as pf:
        writer = csv.writer(pf)
        for i, m in enumerate(metadata_test):
            writer.writerow([m[0], *predicted_y[i].tolist()])

    print(f"writing actual values to {actual_file}...")
    with open(actual_file, "wt") as af:
        writer = csv.writer(af)
        for i, m in enumerate(metadata_test):
            writer.writerow([m[0], *y_test[i].tolist()])

    if model_file is not None:
        print(f"persisting model to {model_file}...")
        with gzip.open(model_file, "wb") as mf:
            pickle.dump(regressor, mf, protocol=pickle.HIGHEST_PROTOCOL)
