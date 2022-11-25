import numpy as np


def get_prediction_indices(doping_levels=None, doping_types=None, temperatures=None):
    """
    TODO add docstring
    :param doping_levels:
    :param doping_types:
    :param temperatures:
    :return:
    """
    indices = []

    if doping_types is not None:
        doping_types = set(doping_types)
        if "p" in doping_types:
            indices.append(np.concatenate([
                np.r_[0:13],    # 1e+16
                np.r_[26:39],   # 1e+17
                np.r_[52:65],   # 1e+18
                np.r_[78:91],   # 1e+19
                np.r_[104:117]  # 1e+20
            ]))

        if "n" in doping_types:
            indices.append(np.concatenate([
                np.r_[13:26],   # 1e+16
                np.r_[39:52],   # 1e+17
                np.r_[65:78],   # 1e+18
                np.r_[91:104],  # 1e+19
                np.r_[117:130]  # 1e+20
            ]))

    if len(indices) == 0:
        indices = np.r_[0:130]  # all indices
    else:
        # concat, dedup, sort
        indices = np.concatenate(indices)
        indices = np.array(sorted(set(indices)))

    return indices
