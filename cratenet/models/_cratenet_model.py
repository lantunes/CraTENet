import numpy as np
from scipy import sparse
from pymatgen import Composition


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
