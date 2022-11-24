from sklearn.ensemble import RandomForestRegressor
from pymatgen import Composition
from matminer.featurizers.composition import Meredig


def create_rf_model(n_estimators=200, n_jobs=4, bootstrap=True, max_depth=110, max_features=36):
    """
    TODO write docstring

    :param n_estimators:
    :param n_jobs:
    :param bootstrap:
    :param max_depth:
    :param max_features:
    :return:
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        bootstrap=bootstrap,
        max_depth=max_depth,
        max_features=max_features
    )


def featurize_comp_for_rf(composition, meredig=None, gap=None):
    """
    TODO write docstring
    :param composition: a composition, either as a string or a pymatgen.Composition
    :param meredig: a Meredig featurizer (it's not cheap to create the Meredig featurizer) (optional)
    :param gap: a float representing the band gap (optional)
    :return: a vector representing the featurized composition
    """
    if meredig is None:
        meredig = Meredig()

    if type(composition) == str:
        composition = Composition(composition)

    meredig_vec = meredig.featurize(composition)

    # drop columns 108 and 109 (apparently range and mean AtomicRadius), which contain NaNs in some records
    meredig_vec = [i for j, i in enumerate(meredig_vec) if j not in [108, 109]]
    if gap is not None:
        meredig_vec = list(meredig_vec) + [gap]

    return meredig_vec
