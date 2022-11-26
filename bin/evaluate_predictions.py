import sys
sys.path.extend([".", ".."])
import argparse
from cratenet.analysis import read_data, get_prediction_indices
from sklearn.metrics import r2_score, mean_absolute_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="?", required=True, type=str,
                        help="path to the output file containing the predictions for the holdout set; "
                             "must be a .csv file")
    parser.add_argument("--actual", nargs="?", required=True, type=str,
                        help="path to the output file containing the true values of the holdout set; "
                             "must be a .csv file")
    parser.add_argument("--doping-level", choices=["1e+16", "1e+17", "1e+18", "1e+19", "1e+20"],
                        nargs="+", required=False,
                        help="the doping level(s) for which to evaluate the predictions")
    parser.add_argument("--doping-type", choices=["n", "p"],
                        nargs="+", required=False,
                        help="the doping type(s) for which to evaluate the predictions")
    parser.add_argument("--temperature", choices=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300],
                        nargs="+", required=False, type=int,
                        help="the temperature(s) for which to evaluate the predictions")
    args = parser.parse_args()

    predictions_path = args.predictions
    actual_path = args.actual
    doping_levels = args.doping_level
    doping_types = args.doping_type
    temperatures = args.temperature

    print(f"reading predictions from {predictions_path}...")
    predicted_data = read_data(predictions_path)

    print(f"reading actual values from {actual_path}...")
    actual_data = read_data(actual_path)

    assert len(predicted_data) == len(actual_data)

    print(f"read predictions and actual values for {len(predicted_data):,} entries")

    print(f"doping types used: {', '.join([dt for dt in doping_types]) if doping_types is not None else 'all'}")
    print(f"doping levels used: {', '.join([dl for dl in doping_levels]) if doping_levels is not None else 'all'}")
    print(f"temperatures used: {', '.join([str(t) for t in temperatures]) if temperatures is not None else 'all'}")

    indices = get_prediction_indices(
        doping_levels=doping_levels,
        doping_types=doping_types,
        temperatures=temperatures)

    predicted_data = predicted_data[:, indices].flatten()
    actual_data = actual_data[:, indices].flatten()

    assert len(actual_data) == len(predicted_data)

    print(f"evaluating {len(predicted_data):,} predictions...")

    mae = mean_absolute_error(actual_data, predicted_data)
    r2 = r2_score(actual_data, predicted_data)

    print(f"MAE: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")
