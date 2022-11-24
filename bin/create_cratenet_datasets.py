import sys
sys.path.extend([".", ".."])
import argparse
from pymatgen import Composition
import gzip
from tqdm import tqdm
from cratenet.dataset import DatasetInputs, read_atom_vectors_from_csv
from cratenet.models import featurize_comp_for_cratenet
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

np.seterr(divide="ignore", invalid="ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seebeck", nargs=2, required=True, type=str,
                        help="the first arg is a path to the deduplicated traces file for the Seebeck (must be a "
                             ".csv.gz file); the second arg is the path the destination file containing the "
                             "dataset (should contain a .csv.gz extension)")
    parser.add_argument("--log10cond", nargs=2, required=True, type=str,
                        help="the first arg is a path to the deduplicated traces file for the electronic conductivity "
                             "(must be a  .csv.gz file); the second arg is the path the destination file containing "
                             "the dataset (should contain a .csv.gz extension)")
    parser.add_argument("--log10pf", nargs=2, required=True, type=str,
                        help="the first arg is a path to the deduplicated traces file for the power factor "
                             "(must be a  .csv.gz file); the second arg is the path the destination file containing "
                             "the dataset (should contain a .csv.gz extension)")
    parser.add_argument("--atom-vectors", nargs="?", required=True, type=str,
                        help="path to atom vectors .csv file: a file containing a list of the supported atoms and the "
                             "components of the corresponding atom vectors; these vectors will be used as the "
                             "input representations for the model; only compounds containing atoms in this list will "
                             "be included in the dataset")
    parser.add_argument("--max-elements", nargs="?", required=False, type=int, default=8,
                        help="the maximum number of elements occurring in a composition in the dataset")
    parser.add_argument("--gaps", nargs="?", required=False, type=str,
                        help="path to the .csv file containing a mapping from composition to corresponding band gap")
    parser.add_argument("--metadata", nargs="?", required=False, type=str,
                        help="path to the .csv file containing a mapping from composition to associated metadata "
                             "(e.g. MP ID)")
    args = parser.parse_args()

    seebeck_traces_file = args.seebeck[0]
    cratenet_seebeck_file = args.seebeck[1]
    cond_traces_file = args.log10cond[0]
    cratenet_log10cond_file = args.log10cond[1]
    pf_traces_file = args.log10pf[0]
    cratenet_log10pf_file = args.log10pf[1]
    max_elements = args.max_elements
    comp_gaps_file = args.gaps
    metadata_file = args.metadata

    print(f"maximum number of elements in a composition: {max_elements}")

    print(f"reading atom vectors from {args.atom_vectors}...")
    atom_dict, atom_vectors = read_atom_vectors_from_csv(args.atom_vectors)
    supported_atoms = atom_dict.keys()
    print("num atoms: %s" % len(supported_atoms))

    print("reading traces files...")
    inputs = DatasetInputs(seebeck_traces_file, cond_traces_file, pf_traces_file, comp_gaps_file, metadata_file)

    print("creating datasets...")
    cratenet_seebeck_dataset = []
    cratenet_log10cond_dataset = []
    cratenet_log10pf_dataset = []
    metadata = []
    formulas = {formula for formula, _, _ in inputs.comp_to_seebeck_traces}
    for formula in tqdm(formulas):
        composition = Composition(formula)
        if any([e.name not in supported_atoms for e in composition.elements]):
            continue

        seebeck_traces_combined = DatasetInputs.combine_traces(formula, inputs.comp_to_seebeck_traces)
        log10cond_traces_combined = DatasetInputs.combine_traces(formula, inputs.comp_to_log10cond_traces)
        log10pf_traces_combined = DatasetInputs.combine_traces(formula, inputs.comp_to_log10pf_traces)

        gap = inputs.comps_to_gaps[formula] if inputs.include_gap else None

        cratenet_input = featurize_comp_for_cratenet(composition, atom_dict, atom_vectors, max_elements, gap=gap)

        cratenet_seebeck_dataset.append([cratenet_input, seebeck_traces_combined])
        cratenet_log10cond_dataset.append([cratenet_input, log10cond_traces_combined])
        cratenet_log10pf_dataset.append([cratenet_input, log10pf_traces_combined])

        metadata.append((formula, inputs.comp_to_metadata[formula] if inputs.has_metadata else ""))

    print(f"number of entries in Seebeck dataset: {len(cratenet_seebeck_dataset):,}")
    print(f"number of entries in log10 electronic conductivity dataset: {len(cratenet_log10cond_dataset):,}")
    print(f"number of entries in log10 PF dataset: {len(cratenet_log10pf_dataset):,}")
    print(f"number of metadata entries: {len(metadata):,}")

    print(f"writing {cratenet_seebeck_file}...")
    with gzip.open(cratenet_seebeck_file, "wb") as f:
        pickle.dump((metadata, cratenet_seebeck_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"writing {cratenet_log10cond_file}...")
    with gzip.open(cratenet_log10cond_file, "wb") as f:
        pickle.dump((metadata, cratenet_log10cond_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"writing {cratenet_log10pf_file}...")
    with gzip.open(cratenet_log10pf_file, "wb") as f:
        pickle.dump((metadata, cratenet_log10pf_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
