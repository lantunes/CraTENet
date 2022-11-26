import csv
import numpy as np


def read_data(fname, expected_vals_len=130):
    """
    TODO add docstring
    :param fname:
    :param expected_vals_len:
    :return:
    """
    data = []
    with open(fname, "rt") as f:
        reader = csv.reader(f)
        for line in reader:
            vals = [float(x) for x in line[1:]]
            assert len(vals) == expected_vals_len
            data.append(vals)
    return np.array(data)


def get_prediction_indices(doping_levels=None, doping_types=None, temperatures=None):
    """
    TODO add docstring
    :param doping_levels:
    :param doping_types:
    :param temperatures:
    :return:
    """

    segments = [
        ("p", "1e+16", np.r_[0:13]),
        ("n", "1e+16", np.r_[13:26]),
        ("p", "1e+17", np.r_[26:39]),
        ("n", "1e+17", np.r_[39:52]),
        ("p", "1e+18", np.r_[52:65]),
        ("n", "1e+18", np.r_[65:78]),
        ("p", "1e+19", np.r_[78:91]),
        ("n", "1e+19", np.r_[91:104]),
        ("p", "1e+20", np.r_[104:117]),
        ("n", "1e+20", np.r_[117:130]),
    ]
    all_segments = np.r_[0:130]
    doping_types = set(doping_types) if doping_types is not None else None
    doping_levels = set(doping_levels) if doping_levels is not None else None
    temperatures = set(temperatures) if temperatures is not None else None

    all_indices = []

    def _include_segment(dop_typ, dop_lev):
        if doping_types is None and doping_levels is None:
            return True
        elif doping_types is None and doping_levels is not None:
            return dop_lev in doping_levels
        elif doping_types is not None and doping_levels is None:
            return dop_typ in doping_types
        elif doping_types is not None and doping_levels is not None:
            return dop_lev in doping_levels and dop_typ in doping_types

    for doping_type, doping_level, indices in segments:
        if _include_segment(doping_type, doping_level):
            all_indices.append(indices)

    if temperatures is not None:
        temp_filter = [int(temp/100)-1 for temp in temperatures]
        for i in range(len(all_indices)):
            all_indices[i] = all_indices[i][temp_filter]

    if len(all_indices) == 0:
        all_indices = all_segments
    else:
        all_indices = np.concatenate(all_indices)
        all_indices = np.array(sorted(all_indices))

    return all_indices
