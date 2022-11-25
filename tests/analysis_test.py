import unittest
import numpy as np
from cratenet.analysis import get_prediction_indices


class TestAnalysis(unittest.TestCase):

    def test_get_prediction_indices_all(self):
        indices = get_prediction_indices()
        np.testing.assert_array_equal(indices, np.r_[0:130])

    def test_get_prediction_indices_p_type(self):
        indices = get_prediction_indices(doping_types=["p"])
        np.testing.assert_array_equal(indices, np.concatenate([
            np.r_[0:13],    # 1e+16
            np.r_[26:39],   # 1e+17
            np.r_[52:65],   # 1e+18
            np.r_[78:91],   # 1e+19
            np.r_[104:117]  # 1e+20
        ]))

    def test_get_prediction_indices_n_type(self):
        indices = get_prediction_indices(doping_types=["n"])
        np.testing.assert_array_equal(indices, np.concatenate([
            np.r_[13:26],   # 1e+16
            np.r_[39:52],   # 1e+17
            np.r_[65:78],   # 1e+18
            np.r_[91:104],  # 1e+19
            np.r_[117:130]  # 1e+20
        ]))

    def test_get_prediction_indices_p_and_n_type(self):
        indices = get_prediction_indices(doping_types=["p", "n"])
        np.testing.assert_array_equal(indices, np.r_[0:130])


if __name__ == '__main__':
    unittest.main()
