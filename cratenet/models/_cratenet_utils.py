import numpy as np
from scipy import sparse
from pymatgen import Composition
from tensorflow.keras import backend as K


def featurize_comp_for_cratenet(comp, dictionary, embeddings, max_elements, gap=None):
    """
    TODO write docstring
    :param comp: a composition, either as a string or a pymatgen.Composition
    :param dictionary:
    :param embeddings:
    :param max_elements:
    :param gap: a float representing the band gap (optional)
    :return:
    """
    if type(comp) == str:
        comp = Composition(comp)

    unscaled_vectors = np.zeros((max_elements, len(embeddings[0])))
    amounts = np.zeros(max_elements)

    for i, e in enumerate(comp.elements):
        unscaled_vectors[i] = np.array(embeddings[dictionary[e.name]])
        amounts[i] = comp.to_reduced_dict[e.name]

    amounts = amounts / sum(amounts)

    matrix = sparse.coo_matrix(unscaled_vectors.tolist())

    if gap is not None:
        return matrix, amounts, gap
    return matrix, amounts


def get_unscaled_mae_metric_for_standard_scaler_nout_robust(target_scaler):
    """
    TODO write docstring
    :param target_scaler:
    :return:
    """
    def unscaled_mae(y_true, y_pred):
        y_pred = y_pred[:, ::2]
        y_true *= K.constant(target_scaler.scale_)
        y_true += K.constant(target_scaler.mean_)

        y_pred *= K.constant(target_scaler.scale_)
        y_pred += K.constant(target_scaler.mean_)
        return K.mean(K.abs(y_true - y_pred), axis=-1)
    return unscaled_mae
