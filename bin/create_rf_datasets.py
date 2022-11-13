import os
from pymatgen import Composition
from skipatom import OneHotVectors
from matminer.featurizers.composition import Meredig
import csv
import gzip
import numpy as np
from tqdm import tqdm

try:
    import cPickle as pickle
except ImportError:
    import pickle
    
SUPPORTED_ATOMS = ["Si", "C", "Pb", "I", "Br", "Cl", "Eu", "O", "Fe", "Sb", "In", "S", "N", "U", "Mn", "Lu", "Se",
                   "Tl", "Hf", "Ir", "Ca", "Ta", "Cr", "K", "Pm", "Mg", "Zn", "Cu", "Sn", "Ti", "B", "W", "P", "H",
                   "Pd", "As", "Co", "Np", "Tc", "Hg", "Pu", "Al", "Tm", "Tb", "Ho", "Nb", "Ge", "Zr", "Cd", "V", "Sr",
                   "Ni", "Rh", "Th", "Na", "Ru", "La", "Re", "Y", "Er", "Ce", "Pt", "Ga", "Li", "Cs", "F", "Ba", "Te",
                   "Mo", "Gd", "Pr", "Bi", "Sc", "Ag", "Rb", "Dy", "Yb", "Nd", "Au", "Os", "Pa", "Sm", "Be", "Ac",
                   "Xe", "Kr"]
EXCLUDED_ATOMS = ["He", "Ar", "Ne"]


def get_all_mpid_to_gaps(all_gaps_file):
    mpid_to_gap = {}
    with open(all_gaps_file, "rt") as f:
        reader = csv.reader(f)
        for line in reader:
            mpid_to_gap[line[0]] = float(line[1])
    return mpid_to_gap


def get_all_mpid_to_ener_per_atom(mp_all_ener_per_atom_file):
    mpid_to_form_e = {}
    with gzip.open(mp_all_ener_per_atom_file, "rt") as f:
        reader = csv.reader(f)
        for line in reader:
            mpid_to_form_e[line[0]] = float(line[2])
    return mpid_to_form_e


def get_mpid_to_traces(traces_file, is_log10=False):
    mpid_to_traces = {}
    with gzip.open(traces_file, "rt") as f:
        reader = csv.reader(f)
        for line in reader:

            if is_log10:
                vals = [np.log10(float(line[i])) for i in range(3, 16)]
                vals = [0.0 if np.isneginf(val) or np.isnan(val) or np.isinf(val) else val for val in vals]
            else:
                vals = [float(line[i]) for i in range(3, 16)]

            key = (line[0], line[1], line[2])
            mpid_to_traces[key] = vals
    return mpid_to_traces


if __name__ == '__main__':

    """
    Inputs
    """
    seebeck_traces_file = "../out/seebeck_traces.csv.gz"
    cond_traces_file = "../out/cond_traces.csv.gz"
    pf_traces_file = "../out/pf_traces.csv.gz"
    # all the formulas of the Ricci database, associated with their MP ID
    all_formulas_file = "../data/ricci_formulas.csv"
    all_gaps_file = "../data/ricci_gaps.csv"  # use the Ricci database gaps
    mp_all_ener_per_atom_file = "../data/mp-2022-03-10-ricci_task_ener_per_atom.csv.gz"
    data_out_dir = "../out/datasets_minpol"
    supported_atoms = SUPPORTED_ATOMS
    excluded_atoms = EXCLUDED_ATOMS

    """
    Outputs
    """
    meredig_seebeck_file = "minpol_meredig_seebeck_nD_nT_nL.pkl.gz"
    meredig_gap_seebeck_file = "minpol_meredig+gap_seebeck_nD_nT_nL.pkl.gz"
    meredig_log10cond_file = "minpol_meredig_log10cond_nD_nT_nL.pkl.gz"
    meredig_gap_log10cond_file = "minpol_meredig+gap_log10cond_nD_nT_nL.pkl.gz"
    meredig_log10pf_file = "minpol_meredig_log10pf_nD_nT_nL.pkl.gz"
    meredig_gap_log10pf_file = "minpol_meredig+gap_log10pf_nD_nT_nL.pkl.gz"


    mpid_to_seebeck_traces = get_mpid_to_traces(seebeck_traces_file)
    mpid_to_log10cond_traces = get_mpid_to_traces(cond_traces_file, is_log10=True)
    mpid_to_log10pf_traces = get_mpid_to_traces(pf_traces_file, is_log10=True)

    mpid_to_gaps = get_all_mpid_to_gaps(all_gaps_file)
    mpid_to_ener_per_atom = get_all_mpid_to_ener_per_atom(mp_all_ener_per_atom_file)

    ohv = OneHotVectors(elems=supported_atoms)

    meredig = Meredig()

    formula_to_ener_per_atom = {}
    formula_to_seebecks = {}
    formula_to_log10conds = {}
    formula_to_log10pfs = {}
    formula_to_gap = {}
    formula_to_mpid = {}
    formula_to_comp = {}
    with open(all_formulas_file, "rt") as in_f:
        reader = csv.reader(in_f)
        for line in tqdm(reader):
            mpid = line[0]
            formula = line[1]

            ener_per_atom = mpid_to_ener_per_atom[mpid]
            composition = Composition(formula)
            if any([e.name in excluded_atoms for e in composition.elements]):
                continue

            seebeck_traces_combined = list(mpid_to_seebeck_traces[(mpid, "p", "1e16")])
            seebeck_traces_combined.extend(mpid_to_seebeck_traces[(mpid, "n", "1e16")])
            seebeck_traces_combined.extend(mpid_to_seebeck_traces[(mpid, "p", "1e17")])
            seebeck_traces_combined.extend(mpid_to_seebeck_traces[(mpid, "n", "1e17")])
            seebeck_traces_combined.extend(mpid_to_seebeck_traces[(mpid, "p", "1e18")])
            seebeck_traces_combined.extend(mpid_to_seebeck_traces[(mpid, "n", "1e18")])
            seebeck_traces_combined.extend(mpid_to_seebeck_traces[(mpid, "p", "1e19")])
            seebeck_traces_combined.extend(mpid_to_seebeck_traces[(mpid, "n", "1e19")])
            seebeck_traces_combined.extend(mpid_to_seebeck_traces[(mpid, "p", "1e20")])
            seebeck_traces_combined.extend(mpid_to_seebeck_traces[(mpid, "n", "1e20")])

            log10cond_traces_combined = list(mpid_to_log10cond_traces[(mpid, "p", "1e16")])
            log10cond_traces_combined.extend(mpid_to_log10cond_traces[(mpid, "n", "1e16")])
            log10cond_traces_combined.extend(mpid_to_log10cond_traces[(mpid, "p", "1e17")])
            log10cond_traces_combined.extend(mpid_to_log10cond_traces[(mpid, "n", "1e17")])
            log10cond_traces_combined.extend(mpid_to_log10cond_traces[(mpid, "p", "1e18")])
            log10cond_traces_combined.extend(mpid_to_log10cond_traces[(mpid, "n", "1e18")])
            log10cond_traces_combined.extend(mpid_to_log10cond_traces[(mpid, "p", "1e19")])
            log10cond_traces_combined.extend(mpid_to_log10cond_traces[(mpid, "n", "1e19")])
            log10cond_traces_combined.extend(mpid_to_log10cond_traces[(mpid, "p", "1e20")])
            log10cond_traces_combined.extend(mpid_to_log10cond_traces[(mpid, "n", "1e20")])

            log10pf_traces_combined = list(mpid_to_log10pf_traces[(mpid, "p", "1e16")])
            log10pf_traces_combined.extend(mpid_to_log10pf_traces[(mpid, "n", "1e16")])
            log10pf_traces_combined.extend(mpid_to_log10pf_traces[(mpid, "p", "1e17")])
            log10pf_traces_combined.extend(mpid_to_log10pf_traces[(mpid, "n", "1e17")])
            log10pf_traces_combined.extend(mpid_to_log10pf_traces[(mpid, "p", "1e18")])
            log10pf_traces_combined.extend(mpid_to_log10pf_traces[(mpid, "n", "1e18")])
            log10pf_traces_combined.extend(mpid_to_log10pf_traces[(mpid, "p", "1e19")])
            log10pf_traces_combined.extend(mpid_to_log10pf_traces[(mpid, "n", "1e19")])
            log10pf_traces_combined.extend(mpid_to_log10pf_traces[(mpid, "p", "1e20")])
            log10pf_traces_combined.extend(mpid_to_log10pf_traces[(mpid, "n", "1e20")])

            if formula not in formula_to_ener_per_atom or ener_per_atom < formula_to_ener_per_atom[formula]:
                # either the formula hasn't been seen yet,
                #  or we have a lower energy polymorph and we'll use this one instead
                formula_to_ener_per_atom[formula] = ener_per_atom
                formula_to_seebecks[formula] = seebeck_traces_combined
                formula_to_log10conds[formula] = log10cond_traces_combined
                formula_to_log10pfs[formula] = log10pf_traces_combined
                formula_to_gap[formula] = mpid_to_gaps[mpid]
                formula_to_mpid[formula] = mpid
                formula_to_comp[formula] = composition

    meredig_seebeck_dataset = []
    meredig_gap_seebeck_dataset = []
    meredig_log10cond_dataset = []
    meredig_gap_log10cond_dataset = []
    meredig_log10pf_dataset = []
    meredig_gap_log10pf_dataset = []
    metadata = []
    for formula in tqdm(formula_to_seebecks):
        composition = formula_to_comp[formula]
        seebeck_traces = formula_to_seebecks[formula]
        log10cond_traces = formula_to_log10conds[formula]
        log10pf_traces = formula_to_log10pfs[formula]
        gap = formula_to_gap[formula]
        mpid = formula_to_mpid[formula]

        meredig_vec = meredig.featurize(composition)  # it's not cheap to create the Meredig vector
        # drop columns 108 and 109 (apparently range and mean AtomicRadius), which contain NaNs in some records
        meredig_vec = [i for j, i in enumerate(meredig_vec) if j not in [108, 109]]
        meredig_vec_with_gap = list(meredig_vec) + [gap]
        meredig_seebeck_dataset.append([meredig_vec, seebeck_traces])
        meredig_gap_seebeck_dataset.append([meredig_vec_with_gap, seebeck_traces])
        meredig_log10cond_dataset.append([meredig_vec, log10cond_traces])
        meredig_gap_log10cond_dataset.append([meredig_vec_with_gap, log10cond_traces])
        meredig_log10pf_dataset.append([meredig_vec, log10pf_traces])
        meredig_gap_log10pf_dataset.append([meredig_vec_with_gap, log10pf_traces])

        metadata.append((formula, mpid))

    print(len(meredig_seebeck_dataset))
    print(len(meredig_gap_seebeck_dataset))
    print(len(meredig_log10cond_dataset))
    print(len(meredig_gap_log10cond_dataset))
    print(len(meredig_log10pf_dataset))
    print(len(meredig_gap_log10pf_dataset))
    print(len(metadata))

    with gzip.open(os.path.join(data_out_dir, meredig_seebeck_file), "wb") as f:
        pickle.dump((metadata, meredig_seebeck_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.open(os.path.join(data_out_dir, meredig_gap_seebeck_file), "wb") as f:
        pickle.dump((metadata, meredig_gap_seebeck_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.open(os.path.join(data_out_dir, meredig_log10cond_file), "wb") as f:
        pickle.dump((metadata, meredig_log10cond_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.open(os.path.join(data_out_dir, meredig_gap_log10cond_file), "wb") as f:
        pickle.dump((metadata, meredig_gap_log10cond_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.open(os.path.join(data_out_dir, meredig_log10pf_file), "wb") as f:
        pickle.dump((metadata, meredig_log10pf_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

    with gzip.open(os.path.join(data_out_dir, meredig_gap_log10pf_file), "wb") as f:
        pickle.dump((metadata, meredig_gap_log10pf_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
