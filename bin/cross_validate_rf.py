import sys
sys.path.extend([".", ".."])
import os
import argparse
import csv
import numpy as np
from cratenet.models import create_rf_model
from cratenet.dataset import load_gzipped_dataset
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="?", required=True, type=str,
                        help="the dataset to use (must be a .csv.gz file)")
    parser.add_argument("--folds", nargs="?", required=False, type=int, default=10,
                        help="the number of folds to use for cross-validation")
    parser.add_argument("--results-dir", required=False, type=str,
                        help="path to the directory where .csv files with the results for each fold will be written")
    parser.add_argument("--seed", nargs="?", required=False, type=int, default=20012022,
                        help="the random seed to use")
    args = parser.parse_args()

    dataset_file = args.dataset
    random_state = args.seed
    results_dir = args.results_dir
    n_folds = args.folds

    if results_dir is not None and not os.path.exists(results_dir):
        print(f"creating non-existent directory: {results_dir}")
        os.makedirs(results_dir)

    print(f"performing {n_folds}-fold cross-validation...")
    print(" ".join(f"model: {create_rf_model()}".replace("\n", "").split()))
    print(f"random seed: {random_state}")

    metadata, X, y = load_gzipped_dataset(dataset_file)
    metadata = np.array(metadata)
    print(f"num examples: {len(X):,}")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold = 0
    maes = []
    r2s = []
    for train, test in kfold.split(X, y):
        fold += 1
        print(f"FOLD {fold}")

        regressor = create_rf_model()

        print("training...")

        regressor.fit(X[train], y[train])

        print("predicting...")

        predicted_y = regressor.predict(X[test])
        y_test = y[test]

        if results_dir is not None:
            predictions_file = os.path.join(results_dir, f"cv_predictions.{fold}.csv")
            actual_file = os.path.join(results_dir, f"cv_actual.{fold}.csv")

            print(f"writing predictions to {predictions_file}...")
            with open(predictions_file, "wt") as pf:
                writer = csv.writer(pf)
                for i, m in enumerate(metadata[test]):
                    writer.writerow([m[0], *predicted_y[i].tolist()])

            print(f"writing actual values to {actual_file}...")
            with open(actual_file, "wt") as af:
                writer = csv.writer(af)
                for i, m in enumerate(metadata[test]):
                    writer.writerow([m[0], *y_test[i].tolist()])

        predicted_y = predicted_y.flatten()
        y_test = y_test.flatten()

        mae = mean_absolute_error(y_test, predicted_y)
        r2 = r2_score(y_test, predicted_y)

        print(f"MAE: {mae:.4}, r2: {r2:.4}")

        maes.append(mae)
        r2s.append(r2)

    print("Fold | MAE      | R2")
    for i in range(len(maes)):
        print(f'{(i + 1): <4} | {"%.4f" % maes[i]: <6} | {"%.4f" % r2s[i]: <6}')
    print(f"mean MAE: {np.mean(maes):.4} ± {np.std(maes):.4}")
    print(f"mean R2: {np.mean(r2s):.4} ± {np.std(r2s):.4}")
