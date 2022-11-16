import csv
import gzip
import numpy as np
from scipy import sparse
from tqdm import tqdm


class DatasetInputs(object):
    def __init__(self, seebeck_file, cond_file, pf_file, gaps_file, metadata_file):
        self.comp_to_seebeck_traces = self.get_comp_to_traces(seebeck_file)
        self.comp_to_log10cond_traces = self.get_comp_to_traces(cond_file, is_log10=True)
        self.comp_to_log10pf_traces = self.get_comp_to_traces(pf_file, is_log10=True)

        self.comps_to_gaps = None
        if gaps_file is not None:
            self.comps_to_gaps = self.get_comps_to_gaps(gaps_file)
        self.include_gap = self.comps_to_gaps is not None

        self.comp_to_metadata = None
        if metadata_file is not None:
            self.comp_to_metadata = self.get_comp_to_metadata(metadata_file)
        self.has_metadata = self.comp_to_metadata is not None

    @staticmethod
    def get_comp_to_traces(traces_file, is_log10=False):
        comp_to_traces = {}
        with gzip.open(traces_file, "rt") as f:
            reader = csv.reader(f)
            for line in tqdm(reader):

                if is_log10:
                    vals = [np.log10(float(line[i])) for i in range(3, 16)]
                    vals = [0.0 if np.isneginf(val) or np.isnan(val) or np.isinf(val) else val for val in vals]
                else:
                    vals = [float(line[i]) for i in range(3, 16)]

                key = (line[0], line[1], line[2])
                comp_to_traces[key] = vals
        return comp_to_traces

    @staticmethod
    def get_comps_to_gaps(gaps_file):
        comp_to_gap = {}
        with open(gaps_file, "rt") as f:
            reader = csv.reader(f)
            for line in reader:
                comp_to_gap[line[0]] = float(line[1])
        return comp_to_gap

    @staticmethod
    def get_comp_to_metadata(metadata_file):
        comp_to_metadata = {}
        with open(metadata_file, "rt") as f:
            reader = csv.reader(f)
            for line in reader:
                comp_to_metadata[line[0]] = line[1]
        return comp_to_metadata

    @staticmethod
    def combine_traces(composition, comp_to_traces):
        traces_combined = list(comp_to_traces[(composition, "p", "1e+16")])
        traces_combined.extend(comp_to_traces[(composition, "n", "1e+16")])
        traces_combined.extend(comp_to_traces[(composition, "p", "1e+17")])
        traces_combined.extend(comp_to_traces[(composition, "n", "1e+17")])
        traces_combined.extend(comp_to_traces[(composition, "p", "1e+18")])
        traces_combined.extend(comp_to_traces[(composition, "n", "1e+18")])
        traces_combined.extend(comp_to_traces[(composition, "p", "1e+19")])
        traces_combined.extend(comp_to_traces[(composition, "n", "1e+19")])
        traces_combined.extend(comp_to_traces[(composition, "p", "1e+20")])
        traces_combined.extend(comp_to_traces[(composition, "n", "1e+20")])
        return traces_combined


def get_transformer_input(comp, dictionary, embeddings, max_elements, gap=None):
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


def read_atom_vectors_from_csv(vectors_file):
    dictionary = {}
    embeddings = []
    with open(vectors_file, "rt") as f:
        reader = csv.reader(f)
        header = next(reader)
        num_components = int(header[-1]) + 1
        for i, line in enumerate(reader):
            atom = line[0]
            vals = [float(v) for v in line[1:]]
            assert len(vals) == num_components
            dictionary[atom] = i
            embeddings.append(vals)
    return dictionary, embeddings
