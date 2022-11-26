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

    def test_get_prediction_indices_all_doping_levels(self):
        indices = get_prediction_indices(
            doping_levels=["1e+16", "1e+17", "1e+18", "1e+19", "1e+20"])
        np.testing.assert_array_equal(indices, np.r_[0:130])

    def test_get_prediction_indices_all_doping_types_and_levels(self):
        indices = get_prediction_indices(
            doping_types=["p", "n"],
            doping_levels=["1e+16", "1e+17", "1e+18", "1e+19", "1e+20"])
        np.testing.assert_array_equal(indices, np.r_[0:130])

    def test_get_prediction_indices_p_doping_type_all_levels(self):
        indices = get_prediction_indices(
            doping_types=["p"],
            doping_levels=["1e+16", "1e+17", "1e+18", "1e+19", "1e+20"])
        np.testing.assert_array_equal(indices, np.concatenate([
            np.r_[0:13],    # 1e+16
            np.r_[26:39],   # 1e+17
            np.r_[52:65],   # 1e+18
            np.r_[78:91],   # 1e+19
            np.r_[104:117]  # 1e+20
        ]))

    def test_get_prediction_indices_n_doping_type_all_levels(self):
        indices = get_prediction_indices(
            doping_types=["n"],
            doping_levels=["1e+16", "1e+17", "1e+18", "1e+19", "1e+20"])
        np.testing.assert_array_equal(indices, np.concatenate([
            np.r_[13:26],   # 1e+16
            np.r_[39:52],   # 1e+17
            np.r_[65:78],   # 1e+18
            np.r_[91:104],  # 1e+19
            np.r_[117:130]  # 1e+20
        ]))

    def test_get_prediction_indices_all_temperatures(self):
        indices = get_prediction_indices(
            temperatures=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300])
        np.testing.assert_array_equal(indices, np.r_[0:130])

    def test_get_prediction_indices_p_doping_type_all_levels_all_temperatures(self):
        indices = get_prediction_indices(
            doping_types=["p"],
            doping_levels=["1e+16", "1e+17", "1e+18", "1e+19", "1e+20"],
            temperatures=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300])
        np.testing.assert_array_equal(indices, np.concatenate([
            np.r_[0:13],    # 1e+16
            np.r_[26:39],   # 1e+17
            np.r_[52:65],   # 1e+18
            np.r_[78:91],   # 1e+19
            np.r_[104:117]  # 1e+20
        ]))

    def test_get_prediction_indices_n_doping_type_all_levels_all_temperatures(self):
        indices = get_prediction_indices(
            doping_types=["n"],
            doping_levels=["1e+16", "1e+17", "1e+18", "1e+19", "1e+20"],
            temperatures=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300])
        np.testing.assert_array_equal(indices, np.concatenate([
            np.r_[13:26],   # 1e+16
            np.r_[39:52],   # 1e+17
            np.r_[65:78],   # 1e+18
            np.r_[91:104],  # 1e+19
            np.r_[117:130]  # 1e+20
        ]))

    def test_get_prediction_indices_all_doping_types_and_levels_and_temperatures(self):
        indices = get_prediction_indices(
            doping_types=["p", "n"],
            doping_levels=["1e+16", "1e+17", "1e+18", "1e+19", "1e+20"],
            temperatures=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300])
        np.testing.assert_array_equal(indices, np.r_[0:130])

    def test_get_prediction_indices_p_doping_type_1e16_level(self):
        indices = get_prediction_indices(
            doping_types=["p"],
            doping_levels=["1e+16"])
        np.testing.assert_array_equal(indices, np.r_[0:13])

    def test_get_prediction_indices_p_doping_type_1e16_level_all_temperatures(self):
        indices = get_prediction_indices(
            doping_types=["p"],
            doping_levels=["1e+16"],
            temperatures=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300])
        np.testing.assert_array_equal(indices, np.r_[0:13])

    def test_get_prediction_indices_n_doping_type_1e16_level(self):
        indices = get_prediction_indices(
            doping_types=["n"],
            doping_levels=["1e+16"])
        np.testing.assert_array_equal(indices, np.r_[13:26])

    def test_get_prediction_indices_n_doping_type_1e16_level_all_temperatures(self):
        indices = get_prediction_indices(
            doping_types=["n"],
            doping_levels=["1e+16"],
            temperatures=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300])
        np.testing.assert_array_equal(indices, np.r_[13:26])

    def test_get_prediction_indices_p_doping_type_1e16_1e18_levels(self):
        indices = get_prediction_indices(
            doping_types=["p"],
            doping_levels=["1e+16", "1e+18"])
        np.testing.assert_array_equal(indices, np.concatenate([
            np.r_[0:13],  # 1e+16
            np.r_[52:65]  # 1e+18
        ]))

    def test_get_prediction_indices_n_doping_type_1e16_1e18_levels(self):
        indices = get_prediction_indices(
            doping_types=["n"],
            doping_levels=["1e+16", "1e+18"])
        np.testing.assert_array_equal(indices, np.concatenate([
            np.r_[13:26],  # 1e+16
            np.r_[65:78]   # 1e+18
        ]))

    def test_get_prediction_indices_all_doping_types_1e16_1e18_levels(self):
        indices = get_prediction_indices(
            doping_types=["p", "n"],
            doping_levels=["1e+16", "1e+18"])
        np.testing.assert_array_equal(indices, np.concatenate([
            np.r_[0:13],   # p 1e+16
            np.r_[13:26],  # n 1e+16
            np.r_[52:65],  # p 1e+18
            np.r_[65:78]   # n 1e+18
        ]))

    def test_get_prediction_indices_p_1e20_600(self):
        indices = get_prediction_indices(
            doping_types=["p"], doping_levels=["1e+20"], temperatures=[600])
        np.testing.assert_array_equal(indices, np.r_[109])

    def test_get_prediction_indices_p_1e20_300_and_600(self):
        indices = get_prediction_indices(
            doping_types=["p"], doping_levels=["1e+20"], temperatures=[300, 600])
        np.testing.assert_array_equal(indices, np.r_[106, 109])

    def test_get_prediction_indices_all_doping_types_1e20_300_and_600(self):
        indices = get_prediction_indices(
            doping_levels=["1e+20"], temperatures=[300, 600])
        np.testing.assert_array_equal(indices, np.r_[106, 109, 119, 122])

    def test_get_prediction_indices_all_doping_types_1e16_and_1e20_300_and_600(self):
        indices = get_prediction_indices(
            doping_levels=["1e+16", "1e+20"], temperatures=[300, 600])
        np.testing.assert_array_equal(indices, np.r_[2, 5, 15, 18, 106, 109, 119, 122])


if __name__ == '__main__':
    unittest.main()
