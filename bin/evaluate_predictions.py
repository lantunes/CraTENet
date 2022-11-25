import sys
sys.path.extend([".", ".."])
import argparse
import csv
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error


def get_indices_for_level_and_temp(doping_type, level, temp):
    temp_idx = int(temp/100) - 1
    if doping_type == "p":
        if level == "1e+16":
            return np.r_[0:13][temp_idx]
        elif level == "1e+17":
            return np.r_[26:39][temp_idx]
        elif level == "1e+18":
            return np.r_[52:65][temp_idx]
        elif level == "1e+19":
            return np.r_[78:91][temp_idx]
        elif level == "1e+20":
            return np.r_[104:117][temp_idx]
        else:
            raise Exception("unsupported level: %s" % level)
    elif doping_type == "n":
        if level == "1e+16":
            return np.r_[13:26][temp_idx]
        elif level == "1e+17":
            return np.r_[39:52][temp_idx]
        elif level == "1e+18":
            return np.r_[65:78][temp_idx]
        elif level == "1e+19":
            return np.r_[91:104][temp_idx]
        elif level == "1e+20":
            return np.r_[117:130][temp_idx]
        else:
            raise Exception("unsupported level: %s" % level)
    else:
        raise Exception("unsupported doping type: %s" % doping_type)


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
                        nargs="+", required=False,
                        help="the temperature(s) for which to evaluate the predictions")
    args = parser.parse_args()

    """
    # evaluate all predictions - if no options are specified 
    # evaluate predictions at 1e+16 only, 1e+17 only, ...
    # evaluate p-type only predictions
    # evaluate n-type only predictions
    # evaluate predictions at p-type @1e+20 only
    # evaluate predictions at n-type @1e+20 only
    # evaluate p-type predictions @1e+20 @600K only
    # evaluate n-type predictions @1e+20 @600K only    
    """

    indices = np.r_[0:130]  # all indices

    # TODO select the indices given the args
