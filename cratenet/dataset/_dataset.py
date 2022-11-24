import csv
import gzip
import numpy as np
from tqdm import tqdm

try:
    import cPickle as pickle
except ImportError:
    import pickle


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


def load_gzipped_dataset(filename):
    with gzip.open(filename, "rb") as f:
        metadata, data = pickle.load(f)
        dataset = np.array(data)
        return metadata, np.array(dataset[:, 0].tolist()), np.array(dataset[:, 1].tolist())
