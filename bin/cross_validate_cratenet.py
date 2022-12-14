import sys
sys.path.append(".")
import argparse
import os
import csv
import numpy as np
from cratenet.dataset import load_gzipped_dataset
from cratenet.models import CraTENet, MultiHeadEarlyStopping, LogMetrics, RobustL1LossMultiOut, RobustL2LossMultiOut, \
    get_unscaled_mae_metric_for_standard_scaler_nout_robust
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-seebeck", nargs="?", required=True, type=str,
                        help="the Seebeck dataset to use (must be a .csv.gz file)")
    parser.add_argument("--dataset-log10cond", nargs="?", required=True, type=str,
                        help="the log10 electrical conductivity dataset to use (must be a .csv.gz file)")
    parser.add_argument("--dataset-log10pf", nargs="?", required=True, type=str,
                        help="the log10 PF dataset to use (must be a .csv.gz file)")
    parser.add_argument("--folds", nargs="?", required=False, type=int, default=10,
                        help="the number of folds to use for cross-validation")
    parser.add_argument("--with-gaps", action="store_true",
                        help="whether gaps should be injected")
    parser.add_argument("--loss", nargs="?", required=False, default="L2", choices=["L1", "L2"],
                        help="the loss to use (either L1 or L2)")
    parser.add_argument("--results-dir", required=False, type=str,
                        help="path to the directory where .csv files with the results for each fold will be written")
    parser.add_argument("--seed", nargs="?", required=False, type=int, default=20012022,
                        help="the random seed to use")
    args = parser.parse_args()

    dataset_seebeck_file = args.dataset_seebeck
    dataset_log10cond_file = args.dataset_log10cond
    dataset_log10pf_file = args.dataset_log10pf
    n_folds = args.folds
    with_gaps = args.with_gaps
    loss_choice = args.loss
    results_dir = args.results_dir
    random_state = args.seed

    if loss_choice == "L2":
        loss = RobustL2LossMultiOut
    elif loss_choice == "L1":
        loss = RobustL1LossMultiOut
    else:
        raise Exception(f"unsupported loss choice: {loss_choice}")

    # TODO these should be parser args
    maxlen = 8
    embed_dim = 200
    num_heads = 4
    ff_dim = 2048
    num_transformer_blocks = 3
    d_model = 512
    n_outputs = 130
    num_epochs = 600
    batch_size = 128
    step_size = 0.0001
    patience = 50
    l2_lambda = 1e-5

    custom_metric = get_unscaled_mae_metric_for_standard_scaler_nout_robust
    holdout_percentage = 1 / n_folds

    if results_dir is not None and not os.path.exists(results_dir):
        print(f"creating non-existent directory: {results_dir}")
        os.makedirs(results_dir)

    task_names = ["seebeck", "log10cond", "log10pf"]
    # each entry contains the target dataset and its corresponding loss weight
    dataset_files = [(dataset_seebeck_file, 1.0),
                     (dataset_log10cond_file, 1.0),
                     (dataset_log10pf_file, 1.0)]

    n_heads = len(dataset_files)
    n_extra_in = 1 if with_gaps else 0

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # load a dataset, just to get the k-fold indices
    _, X_, y_ = load_gzipped_dataset(dataset_files[0][0])

    fold = 0
    maes = {task_name: [] for task_name in task_names}
    r2s = {task_name: [] for task_name in task_names}
    for A, test in kfold.split(X_, y_):

        fold += 1
        print("FOLD %s" % fold)

        loss_weights = []
        target_scalers = []
        X_A_atoms = None
        X_A_amounts = None
        X_A_gaps = None
        X_test_atoms = None
        X_test_amounts = None
        X_test_gaps = None
        X_train_atoms = None
        X_train_amounts = None
        X_train_gaps = None
        X_val_atoms = None
        X_val_amounts = None
        X_val_gaps = None
        metadata_test_ = None
        y_As = []
        y_trains = []
        y_vals = []
        y_tests = []
        for dataset_file, loss_weight in dataset_files:
            loss_weights.append(loss_weight)

            metadata, X, y = load_gzipped_dataset(dataset_file)

            metadata = np.array(metadata)
            X = np.array(X)
            target_scaler = StandardScaler()
            target_scaler.fit(np.array(y).reshape(-1, 1))
            target_scalers.append(target_scaler)
            y = target_scaler.transform(np.array(y).reshape(-1, 1)).reshape(-1, n_outputs)
            if n_outputs == 1:
                y = y.flatten()

            X_A = X[A]
            X_test = X[test]
            y_A = y[A]
            y_test = y[test]
            metadata_A = metadata[A]
            metadata_test = metadata[test]

            X_train, X_val, y_train, y_val = \
                train_test_split(X_A, y_A, random_state=random_state, shuffle=True, test_size=holdout_percentage)

            if X_A_atoms is None:
                X_A_atoms = np.array([x[0].toarray().tolist() for x in X_A])
                X_A_amounts = np.array([x[1] for x in X_A])
                X_A_gaps = np.array([x[2] for x in X_A]).reshape(-1, 1) if with_gaps else None
                X_test_atoms = np.array([x[0].toarray().tolist() for x in X_test])
                X_test_amounts = np.array([x[1] for x in X_test])
                X_test_gaps = np.array([x[2] for x in X_test]).reshape(-1, 1) if with_gaps else None
                X_train_atoms = np.array([x[0].toarray().tolist() for x in X_train])
                X_train_amounts = np.array([x[1] for x in X_train])
                X_train_gaps = np.array([x[2] for x in X_train]).reshape(-1, 1) if with_gaps else None
                X_val_atoms = np.array([x[0].toarray().tolist() for x in X_val])
                X_val_amounts = np.array([x[1] for x in X_val])
                X_val_gaps = np.array([x[2] for x in X_val]).reshape(-1, 1) if with_gaps else None
                metadata_test_ = metadata_test

            y_As.append(y_A)
            y_trains.append(y_train)
            y_vals.append(y_val)
            y_tests.append(y_test)

        y_As = [np.array(y_a) for y_a in y_As]
        y_trains = [np.array(y_t) for y_t in y_trains]
        y_vals = [np.array(y_v) for y_v in y_vals]
        y_tests = [np.array(y_t) for y_t in y_tests]

        print("training to determine optimal number of epochs...")
        log_metrics = LogMetrics(current_fold=fold)
        log_file = f"train-val-results.{fold}.csv"
        print(f"logging training validation results to {log_file}")
        csv_logger = CSVLogger(log_file, separator=",", append=True)
        early_stopping = MultiHeadEarlyStopping(
            monitor=["val_regr%s_mae" % (i+1) for i in range(n_heads)],
            weights=loss_weights,
            operator=np.less,
            loss_init=np.Inf,
            patience=patience)
        model = CraTENet(maxlen=maxlen, embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
                         num_transformer_blocks=num_transformer_blocks, d_model=d_model, n_heads=n_heads,
                         n_outputs=n_outputs, n_extra_in=n_extra_in)
        train_X = [X_train_atoms, X_train_amounts]
        if with_gaps:
            train_X.append(X_train_gaps)
        val_X = [X_val_atoms, X_val_amounts]
        if with_gaps:
            val_X.append(X_val_gaps)
        model.train(train_X, y_trains, val_X, y_vals,
                    num_epochs=num_epochs, batch_size=batch_size, step_size=step_size, loss=loss,
                    custom_metrics=[[custom_metric(s)] for s in target_scalers],
                    callbacks=[log_metrics, csv_logger, early_stopping])

        n_epochs_retrain = (early_stopping.stopped_epoch+1) - patience if early_stopping.stopped_epoch != 0 else num_epochs

        print("retraining on all training set for %s epochs..." % n_epochs_retrain)
        log_file = f"retrain-results.{fold}.csv"
        print(f"logging retraining results to {log_file}")
        csv_logger = CSVLogger(log_file, separator=",", append=True)
        final_model = CraTENet(maxlen=maxlen, embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim,
                               num_transformer_blocks=num_transformer_blocks, d_model=d_model, n_heads=n_heads,
                               n_outputs=n_outputs, n_extra_in=n_extra_in)
        A_X = [X_A_atoms, X_A_amounts]
        if with_gaps:
            A_X.append(X_A_gaps)
        final_model.train(A_X, y_As, None, None,
                          num_epochs=n_epochs_retrain, batch_size=batch_size, step_size=step_size, loss=loss,
                          custom_metrics=[[custom_metric(s)] for s in target_scalers],
                          callbacks=[log_metrics, csv_logger])

        test_X = [X_test_atoms, X_test_amounts]
        if with_gaps:
            test_X.append(X_test_gaps)
        all_test_predictions = final_model.predict(test_X)

        for i, test_predictions in enumerate(all_test_predictions):
            target_scaler = target_scalers[i]
            task_name = task_names[i]
            # get every other value, since the first value in a pair is the mean and the second is the variance
            test_predictions = test_predictions[:, ::2]

            # unscale the predictions
            test_predictions_unscaled = target_scaler.inverse_transform(test_predictions)
            test_actual_unscaled = target_scaler.inverse_transform(y_tests[i])

            if results_dir is not None:
                predictions_file = os.path.join(results_dir, f"cv_predictions.{task_name}.{fold}.csv")
                actual_file = os.path.join(results_dir, f"cv_actual.{task_name}.{fold}.csv")

                print(f"writing predictions to {predictions_file}...")
                with open(predictions_file, "wt") as pf:
                    writer = csv.writer(pf)
                    for i, m in enumerate(metadata_test_):
                        writer.writerow([m[0], *test_predictions_unscaled[i].tolist()])

                print(f"writing actual values to {actual_file}...")
                with open(actual_file, "wt") as af:
                    writer = csv.writer(af)
                    for i, m in enumerate(metadata_test_):
                        writer.writerow([m[0], *test_actual_unscaled[i].tolist()])

            test_predictions_unscaled = test_predictions_unscaled.flatten()
            test_actual_unscaled = test_actual_unscaled.flatten()
            test_r2 = r2_score(test_actual_unscaled, test_predictions_unscaled)
            test_mae = mean_absolute_error(test_actual_unscaled, test_predictions_unscaled)

            print(f"{task_name}: MAE: {test_mae:.4}, r2: {test_r2:.4}")

            maes[task_name].append(test_mae)
            r2s[task_name].append(test_r2)

    for task_name in task_names:
        print(f"results for {task_name}...")
        print("Fold | MAE      | R2")
        for i in range(len(maes[task_name])):
            print(f'{(i + 1): <4} | {"%.4f" % maes[task_name][i]: <6} | {"%.4f" % r2s[task_name][i]: <6}')
        print(f"mean MAE: {np.mean(maes[task_name]):.4} ± {np.std(maes[task_name]):.4}")
        print(f"mean R2: {np.mean(r2s[task_name]):.4} ± {np.std(r2s[task_name]):.4}")
