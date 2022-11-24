import sys
sys.path.extend([".", ".."])
import argparse
from pymatgen import Composition
from matminer.featurizers.composition import Meredig
from cratenet.dataset import DatasetInputs
from cratenet.models import featurize_comp_for_rf
import gzip
import numpy as np
from tqdm import tqdm

try:
    import cPickle as pickle
except ImportError:
    import pickle

np.seterr(divide="ignore", invalid="ignore")
    
ATOMS = ["Si", "C", "Pb", "I", "Br", "Cl", "Eu", "O", "Fe", "Sb", "In", "S", "N", "U", "Mn", "Lu", "Se", "Tl", "Hf",
         "Ir", "Ca", "Ta", "Cr", "K", "Pm", "Mg", "Zn", "Cu", "Sn", "Ti", "B", "W", "P", "H", "Pd", "As", "Co", "Np",
         "Tc", "Hg", "Pu", "Al", "Tm", "Tb", "Ho", "Nb", "Ge", "Zr", "Cd", "V", "Sr", "Ni", "Rh", "Th", "Na", "Ru",
         "La", "Re", "Y", "Er", "Ce", "Pt", "Ga", "Li", "Cs", "F", "Ba", "Te", "Mo", "Gd", "Pr", "Bi", "Sc", "Ag", "Rb",
         "Dy", "Yb", "Nd", "Au", "Os", "Pa", "Sm", "Be", "Ac", "Xe", "Kr"]


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
    parser.add_argument("--gaps", nargs="?", required=False, type=str,
                        help="path to the .csv file containing a mapping from composition to corresponding band gap")
    parser.add_argument("--metadata", nargs="?", required=False, type=str,
                        help="path to the .csv file containing a mapping from composition to associated metadata "
                             "(e.g. MP ID)")
    parser.add_argument("--atoms", nargs="?", required=False, type=str,
                        help="path to atoms file: a file containing a list of the supported atoms, "
                             "one atom per line; only compounds containing atoms in this list will be included "
                             "in the dataset; a default set of atoms will be used if this argument is not provided")
    args = parser.parse_args()

    seebeck_traces_file = args.seebeck[0]
    meredig_seebeck_file = args.seebeck[1]
    cond_traces_file = args.log10cond[0]
    meredig_log10cond_file = args.log10cond[1]
    pf_traces_file = args.log10pf[0]
    meredig_log10pf_file = args.log10pf[1]
    comp_gaps_file = args.gaps
    metadata_file = args.metadata

    if args.atoms is not None:
        print(f"loading supported atoms from {args.atoms} ...")
        with open(args.atoms, "rt") as f:
            atoms = [line.strip() for line in f.readlines()]
    else:
        atoms = ATOMS
    supported_atoms = set(atoms)

    print("reading traces files...")
    inputs = DatasetInputs(seebeck_traces_file, cond_traces_file, pf_traces_file, comp_gaps_file, metadata_file)

    meredig = Meredig()

    print("creating datasets...")
    meredig_seebeck_dataset = []
    meredig_log10cond_dataset = []
    meredig_log10pf_dataset = []
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

        meredig_vec = featurize_comp_for_rf(composition, meredig=meredig, gap=gap)

        meredig_seebeck_dataset.append([meredig_vec, seebeck_traces_combined])
        meredig_log10cond_dataset.append([meredig_vec, log10cond_traces_combined])
        meredig_log10pf_dataset.append([meredig_vec, log10pf_traces_combined])

        metadata.append((formula, inputs.comp_to_metadata[formula] if inputs.has_metadata else ""))

    print(f"number of entries in Seebeck dataset: {len(meredig_seebeck_dataset):,}")
    print(f"number of entries in log10 electronic conductivity dataset: {len(meredig_log10cond_dataset):,}")
    print(f"number of entries in log10 PF dataset: {len(meredig_log10pf_dataset):,}")
    print(f"number of metadata entries: {len(metadata):,}")

    print(f"writing {meredig_seebeck_file}...")
    with gzip.open(meredig_seebeck_file, "wb") as f:
        pickle.dump((metadata, meredig_seebeck_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"writing {meredig_log10cond_file}...")
    with gzip.open(meredig_log10cond_file, "wb") as f:
        pickle.dump((metadata, meredig_log10cond_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"writing {meredig_log10pf_file}...")
    with gzip.open(meredig_log10pf_file, "wb") as f:
        pickle.dump((metadata, meredig_log10pf_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
