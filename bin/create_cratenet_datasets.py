# import os
# from skipatom import SkipAtomInducedModel, sum_pool
# from pymatgen import Composition
# from matminer.featurizers.composition import Meredig
# from lib import OneHotVectors, get_all_mpid_to_seebeck_traces, get_all_mpid_to_log10cond_traces, get_all_mpid_to_gaps, \
#     get_all_mpid_to_ener_per_atom, get_all_mpid_to_log10pf_traces, to_bravais_lattice_type_ohe, get_all_mpid_to_struct, \
#     get_all_mpid_to_log10S2_traces
# import csv
# import gzip
# from tqdm import tqdm
# from scipy import sparse
# import numpy as np
#
# try:
#     import cPickle as pickle
# except ImportError:
#     import pickle
#
#
# def get_sum_pooled_vector(comp, dictionary, embeddings, gap=None):
#     vec = sum_pool(comp, dictionary, embeddings)
#     if gap is not None:
#         vec.extend([gap])
#     return vec
#
#
# def get_fren_transformer_input(comp, dictionary, embeddings, gap=None, struct=None):
#     # the largest formula has 8 elements
#     unscaled_vectors = np.zeros((8, len(embeddings[0])))
#     amounts = np.zeros(8)
#
#     for i, e in enumerate(comp.elements):
#         unscaled_vectors[i] = np.array(embeddings[dictionary[e.name]])
#         amounts[i] = comp.to_reduced_dict[e.name]
#
#     amounts = amounts / sum(amounts)
#
#     matrix = sparse.coo_matrix(unscaled_vectors.tolist())
#
#     if gap is not None and struct is None:
#         return matrix, amounts, gap
#     if gap is None and struct is not None:
#         return matrix, amounts, struct
#     if gap is not None and struct is not None:
#         return matrix, amounts, gap, struct
#     return matrix, amounts
#
#
# def get_struct_vec(struct):
#     spacegroup, volume, nsites = struct
#     vec = to_bravais_lattice_type_ohe(spacegroup)
#     # max volume for the Ricci database is 6901.60649 and min is 7.8146
#     normalized_volume = (volume - 7) / (7000 - 7)
#     # max nsites for the Ricci database is 200 and min is 1
#     normalized_nsites = (nsites - 1) / (200 - 1)
#     vec = np.concatenate([vec, [normalized_volume, normalized_nsites]])
#     return vec.tolist()
#
#
# """NOTE: 'minpol' -> minimum energy polymorph"""
# if __name__ == '__main__':
#     skipatom_dim = 200
#
#     all_formulas_file = "../out/all_formulas.csv"  # all the formulas of the Ricci database, associated with their MP ID
#
#     all_seebeck_p_traces_file_1e16 = "../out/all_seebeck_trace_p_multiT_1e+16_GGA+U.csv.gz"
#     all_seebeck_p_traces_file_1e17 = "../out/all_seebeck_trace_p_multiT_1e+17_GGA+U.csv.gz"
#     all_seebeck_p_traces_file_1e18 = "../out/all_seebeck_trace_p_multiT_1e+18_GGA+U.csv.gz"
#     all_seebeck_p_traces_file_1e19 = "../out/all_seebeck_trace_p_multiT_1e+19_GGA+U.csv.gz"
#     all_seebeck_p_traces_file_1e20 = "../out/all_seebeck_trace_p_multiT_1e+20_GGA+U.csv.gz"
#
#     all_cond_p_traces_file_1e16 = "../out/all_cond_trace_p_multiT_1e+16_GGA+U.csv.gz"
#     all_cond_p_traces_file_1e17 = "../out/all_cond_trace_p_multiT_1e+17_GGA+U.csv.gz"
#     all_cond_p_traces_file_1e18 = "../out/all_cond_trace_p_multiT_1e+18_GGA+U.csv.gz"
#     all_cond_p_traces_file_1e19 = "../out/all_cond_trace_p_multiT_1e+19_GGA+U.csv.gz"
#     all_cond_p_traces_file_1e20 = "../out/all_cond_trace_p_multiT_1e+20_GGA+U.csv.gz"
#
#     all_pf_p_traces_file_1e16 = "../out/all_pf_trace_p_multiT_1e+16_GGA+U.csv.gz"
#     all_pf_p_traces_file_1e17 = "../out/all_pf_trace_p_multiT_1e+17_GGA+U.csv.gz"
#     all_pf_p_traces_file_1e18 = "../out/all_pf_trace_p_multiT_1e+18_GGA+U.csv.gz"
#     all_pf_p_traces_file_1e19 = "../out/all_pf_trace_p_multiT_1e+19_GGA+U.csv.gz"
#     all_pf_p_traces_file_1e20 = "../out/all_pf_trace_p_multiT_1e+20_GGA+U.csv.gz"
#
#     all_seebeck_n_traces_file_1e16 = "../out/all_seebeck_trace_n_multiT_1e+16_GGA+U.csv.gz"
#     all_seebeck_n_traces_file_1e17 = "../out/all_seebeck_trace_n_multiT_1e+17_GGA+U.csv.gz"
#     all_seebeck_n_traces_file_1e18 = "../out/all_seebeck_trace_n_multiT_1e+18_GGA+U.csv.gz"
#     all_seebeck_n_traces_file_1e19 = "../out/all_seebeck_trace_n_multiT_1e+19_GGA+U.csv.gz"
#     all_seebeck_n_traces_file_1e20 = "../out/all_seebeck_trace_n_multiT_1e+20_GGA+U.csv.gz"
#
#     all_cond_n_traces_file_1e16 = "../out/all_cond_trace_n_multiT_1e+16_GGA+U.csv.gz"
#     all_cond_n_traces_file_1e17 = "../out/all_cond_trace_n_multiT_1e+17_GGA+U.csv.gz"
#     all_cond_n_traces_file_1e18 = "../out/all_cond_trace_n_multiT_1e+18_GGA+U.csv.gz"
#     all_cond_n_traces_file_1e19 = "../out/all_cond_trace_n_multiT_1e+19_GGA+U.csv.gz"
#     all_cond_n_traces_file_1e20 = "../out/all_cond_trace_n_multiT_1e+20_GGA+U.csv.gz"
#
#     all_pf_n_traces_file_1e16 = "../out/all_pf_trace_n_multiT_1e+16_GGA+U.csv.gz"
#     all_pf_n_traces_file_1e17 = "../out/all_pf_trace_n_multiT_1e+17_GGA+U.csv.gz"
#     all_pf_n_traces_file_1e18 = "../out/all_pf_trace_n_multiT_1e+18_GGA+U.csv.gz"
#     all_pf_n_traces_file_1e19 = "../out/all_pf_trace_n_multiT_1e+19_GGA+U.csv.gz"
#     all_pf_n_traces_file_1e20 = "../out/all_pf_trace_n_multiT_1e+20_GGA+U.csv.gz"
#
#     all_gaps_file = "../out/all_gap_GGA+U.csv"  # use the Ricci database gaps
#     mp_all_ener_per_atom_file = "../out/mp-2022-03-10-ricci_task_ener_per_atom.csv.gz"
#     all_struct_file = "../out/all_struct_GGA+U.csv"
#
#     data_out_dir = "../out/datasets_minpol"
#
#     skipatom_sum_seebeck_file = "minpol_skipatom%s_sum_seebeck_nD_nT_nL.pkl.gz" % skipatom_dim
#     skipatom_sum_gap_seebeck_file = "minpol_skipatom%s_sum+gap_seebeck_nD_nT_nL.pkl.gz" % skipatom_dim
#     skipatom_sum_log10cond_file = "minpol_skipatom%s_sum_log10cond_nD_nT_nL.pkl.gz" % skipatom_dim
#     skipatom_sum_gap_log10cond_file = "minpol_skipatom%s_sum+gap_log10cond_nD_nT_nL.pkl.gz" % skipatom_dim
#     fren_transformer_seebeck_file = "minpol_skipatom%s_fren_transformer_seebeck_nD_nT_nL.pkl.gz" % skipatom_dim
#     fren_transformer_gap_seebeck_file = "minpol_skipatom%s_fren+gap_transformer_seebeck_nD_nT_nL.pkl.gz" % skipatom_dim
#     fren_transformer_log10cond_file = "minpol_skipatom%s_fren_transformer_log10cond_nD_nT_nL.pkl.gz" % skipatom_dim
#     fren_transformer_gap_log10cond_file = "minpol_skipatom%s_fren+gap_transformer_log10cond_nD_nT_nL.pkl.gz" % skipatom_dim
#     meredig_seebeck_file = "minpol_meredig_seebeck_nD_nT_nL.pkl.gz"
#     meredig_gap_seebeck_file = "minpol_meredig+gap_seebeck_nD_nT_nL.pkl.gz"
#     meredig_log10cond_file = "minpol_meredig_log10cond_nD_nT_nL.pkl.gz"
#     meredig_gap_log10cond_file = "minpol_meredig+gap_log10cond_nD_nT_nL.pkl.gz"
#     fren_transformer_gap_struct_seebeck_file = "minpol_skipatom%s_fren+gap+struct_transformer_seebeck_nD_nT_nL.pkl.gz" % skipatom_dim
#     fren_transformer_gap_struct_log10cond_file = "minpol_skipatom%s_fren+gap+struct_transformer_log10cond_nD_nT_nL.pkl.gz" % skipatom_dim
#     meredig_gap_struct_seebeck_file = "minpol_meredig+gap+struct_seebeck_nD_nT_nL.pkl.gz"
#     meredig_gap_struct_log10cond_file = "minpol_meredig+gap+struct_log10cond_nD_nT_nL.pkl.gz"
#     fren_transformer_log10pf_file = "minpol_skipatom%s_fren_transformer_log10pf_nD_nT_nL.pkl.gz" % skipatom_dim
#     fren_transformer_gap_log10pf_file = "minpol_skipatom%s_fren+gap_transformer_log10pf_nD_nT_nL.pkl.gz" % skipatom_dim
#     fren_transformer_gap_struct_log10pf_file = "minpol_skipatom%s_fren+gap+struct_transformer_log10pf_nD_nT_nL.pkl.gz" % skipatom_dim
#     meredig_log10pf_file = "minpol_meredig_log10pf_nD_nT_nL.pkl.gz"
#     meredig_gap_log10pf_file = "minpol_meredig+gap_log10pf_nD_nT_nL.pkl.gz"
#     meredig_gap_struct_log10pf_file = "minpol_meredig+gap+struct_log10pf_nD_nT_nL.pkl.gz"
#     fren_transformer_log10S2_file = "minpol_skipatom%s_fren_transformer_log10S2_nD_nT_nL.pkl.gz" % skipatom_dim
#     fren_transformer_gap_log10S2_file = "minpol_skipatom%s_fren+gap_transformer_log10S2_nD_nT_nL.pkl.gz" % skipatom_dim
#     meredig_log10S2_file = "minpol_meredig_log10S2_nD_nT_nL.pkl.gz"
#     meredig_gap_log10S2_file = "minpol_meredig+gap_log10S2_nD_nT_nL.pkl.gz"
#
#     mpid_to_p_seebeck_traces_1e16 = get_all_mpid_to_seebeck_traces(all_seebeck_p_traces_file_1e16)
#     mpid_to_p_seebeck_traces_1e17 = get_all_mpid_to_seebeck_traces(all_seebeck_p_traces_file_1e17)
#     mpid_to_p_seebeck_traces_1e18 = get_all_mpid_to_seebeck_traces(all_seebeck_p_traces_file_1e18)
#     mpid_to_p_seebeck_traces_1e19 = get_all_mpid_to_seebeck_traces(all_seebeck_p_traces_file_1e19)
#     mpid_to_p_seebeck_traces_1e20 = get_all_mpid_to_seebeck_traces(all_seebeck_p_traces_file_1e20)
#
#     mpid_to_p_log10cond_traces_1e16 = get_all_mpid_to_log10cond_traces(all_cond_p_traces_file_1e16)
#     mpid_to_p_log10cond_traces_1e17 = get_all_mpid_to_log10cond_traces(all_cond_p_traces_file_1e17)
#     mpid_to_p_log10cond_traces_1e18 = get_all_mpid_to_log10cond_traces(all_cond_p_traces_file_1e18)
#     mpid_to_p_log10cond_traces_1e19 = get_all_mpid_to_log10cond_traces(all_cond_p_traces_file_1e19)
#     mpid_to_p_log10cond_traces_1e20 = get_all_mpid_to_log10cond_traces(all_cond_p_traces_file_1e20)
#
#     mpid_to_p_log10pf_traces_1e16 = get_all_mpid_to_log10pf_traces(all_pf_p_traces_file_1e16)
#     mpid_to_p_log10pf_traces_1e17 = get_all_mpid_to_log10pf_traces(all_pf_p_traces_file_1e17)
#     mpid_to_p_log10pf_traces_1e18 = get_all_mpid_to_log10pf_traces(all_pf_p_traces_file_1e18)
#     mpid_to_p_log10pf_traces_1e19 = get_all_mpid_to_log10pf_traces(all_pf_p_traces_file_1e19)
#     mpid_to_p_log10pf_traces_1e20 = get_all_mpid_to_log10pf_traces(all_pf_p_traces_file_1e20)
#
#     mpid_to_p_log10S2_traces_1e16 = get_all_mpid_to_log10S2_traces(all_seebeck_p_traces_file_1e16)
#     mpid_to_p_log10S2_traces_1e17 = get_all_mpid_to_log10S2_traces(all_seebeck_p_traces_file_1e17)
#     mpid_to_p_log10S2_traces_1e18 = get_all_mpid_to_log10S2_traces(all_seebeck_p_traces_file_1e18)
#     mpid_to_p_log10S2_traces_1e19 = get_all_mpid_to_log10S2_traces(all_seebeck_p_traces_file_1e19)
#     mpid_to_p_log10S2_traces_1e20 = get_all_mpid_to_log10S2_traces(all_seebeck_p_traces_file_1e20)
#
#     mpid_to_n_seebeck_traces_1e16 = get_all_mpid_to_seebeck_traces(all_seebeck_n_traces_file_1e16)
#     mpid_to_n_seebeck_traces_1e17 = get_all_mpid_to_seebeck_traces(all_seebeck_n_traces_file_1e17)
#     mpid_to_n_seebeck_traces_1e18 = get_all_mpid_to_seebeck_traces(all_seebeck_n_traces_file_1e18)
#     mpid_to_n_seebeck_traces_1e19 = get_all_mpid_to_seebeck_traces(all_seebeck_n_traces_file_1e19)
#     mpid_to_n_seebeck_traces_1e20 = get_all_mpid_to_seebeck_traces(all_seebeck_n_traces_file_1e20)
#
#     mpid_to_n_log10cond_traces_1e16 = get_all_mpid_to_log10cond_traces(all_cond_n_traces_file_1e16)
#     mpid_to_n_log10cond_traces_1e17 = get_all_mpid_to_log10cond_traces(all_cond_n_traces_file_1e17)
#     mpid_to_n_log10cond_traces_1e18 = get_all_mpid_to_log10cond_traces(all_cond_n_traces_file_1e18)
#     mpid_to_n_log10cond_traces_1e19 = get_all_mpid_to_log10cond_traces(all_cond_n_traces_file_1e19)
#     mpid_to_n_log10cond_traces_1e20 = get_all_mpid_to_log10cond_traces(all_cond_n_traces_file_1e20)
#
#     mpid_to_n_log10pf_traces_1e16 = get_all_mpid_to_log10pf_traces(all_pf_n_traces_file_1e16)
#     mpid_to_n_log10pf_traces_1e17 = get_all_mpid_to_log10pf_traces(all_pf_n_traces_file_1e17)
#     mpid_to_n_log10pf_traces_1e18 = get_all_mpid_to_log10pf_traces(all_pf_n_traces_file_1e18)
#     mpid_to_n_log10pf_traces_1e19 = get_all_mpid_to_log10pf_traces(all_pf_n_traces_file_1e19)
#     mpid_to_n_log10pf_traces_1e20 = get_all_mpid_to_log10pf_traces(all_pf_n_traces_file_1e20)
#
#     mpid_to_n_log10S2_traces_1e16 = get_all_mpid_to_log10S2_traces(all_seebeck_n_traces_file_1e16)
#     mpid_to_n_log10S2_traces_1e17 = get_all_mpid_to_log10S2_traces(all_seebeck_n_traces_file_1e17)
#     mpid_to_n_log10S2_traces_1e18 = get_all_mpid_to_log10S2_traces(all_seebeck_n_traces_file_1e18)
#     mpid_to_n_log10S2_traces_1e19 = get_all_mpid_to_log10S2_traces(all_seebeck_n_traces_file_1e19)
#     mpid_to_n_log10S2_traces_1e20 = get_all_mpid_to_log10S2_traces(all_seebeck_n_traces_file_1e20)
#
#     mpid_to_gaps = get_all_mpid_to_gaps(all_gaps_file)
#     mpid_to_ener_per_atom = get_all_mpid_to_ener_per_atom(mp_all_ener_per_atom_file)
#     mpid_to_struct = get_all_mpid_to_struct(all_struct_file)
#
#     pairs = "../out/mp_2020_10_09.dim%s.keras.model" % skipatom_dim
#     td = "../out/mp_2020_10_09.training.data"
#     model = SkipAtomInducedModel.load(pairs, td, min_count=2e7, top_n=5)
#     skipatom_vectors = model.vectors
#     skipatom_dictionary = model.dictionary
#     excluded_atoms = ["He", "Ar", "Ne"]
#
#     atoms = [z[0] for z in sorted(model.dictionary.items(), key=lambda item: item[1])]
#     print("num atoms: %s" % len(atoms))
#
#     ohv = OneHotVectors(elems=atoms)
#
#     meredig = Meredig()
#
#     formula_to_ener_per_atom = {}
#     formula_to_seebecks = {}
#     formula_to_log10conds = {}
#     formula_to_log10pfs = {}
#     formula_to_log10S2s = {}
#     formula_to_gap = {}
#     formula_to_mpid = {}
#     formula_to_comp = {}
#     formula_to_struct = {}
#     with open(all_formulas_file, "rt") as in_f:
#         reader = csv.reader(in_f)
#         for line in tqdm(reader):
#             mpid = line[0]
#             formula = line[1]
#
#             ener_per_atom = mpid_to_ener_per_atom[mpid]
#             composition = Composition(formula)
#             if any([e.name in excluded_atoms for e in composition.elements]):
#                 continue
#
#             seebeck_traces_combined = list(mpid_to_p_seebeck_traces_1e16[mpid])
#             seebeck_traces_combined.extend(mpid_to_n_seebeck_traces_1e16[mpid])
#             seebeck_traces_combined.extend(mpid_to_p_seebeck_traces_1e17[mpid])
#             seebeck_traces_combined.extend(mpid_to_n_seebeck_traces_1e17[mpid])
#             seebeck_traces_combined.extend(mpid_to_p_seebeck_traces_1e18[mpid])
#             seebeck_traces_combined.extend(mpid_to_n_seebeck_traces_1e18[mpid])
#             seebeck_traces_combined.extend(mpid_to_p_seebeck_traces_1e19[mpid])
#             seebeck_traces_combined.extend(mpid_to_n_seebeck_traces_1e19[mpid])
#             seebeck_traces_combined.extend(mpid_to_p_seebeck_traces_1e20[mpid])
#             seebeck_traces_combined.extend(mpid_to_n_seebeck_traces_1e20[mpid])
#
#             log10cond_traces_combined = list(mpid_to_p_log10cond_traces_1e16[mpid])
#             log10cond_traces_combined.extend(mpid_to_n_log10cond_traces_1e16[mpid])
#             log10cond_traces_combined.extend(mpid_to_p_log10cond_traces_1e17[mpid])
#             log10cond_traces_combined.extend(mpid_to_n_log10cond_traces_1e17[mpid])
#             log10cond_traces_combined.extend(mpid_to_p_log10cond_traces_1e18[mpid])
#             log10cond_traces_combined.extend(mpid_to_n_log10cond_traces_1e18[mpid])
#             log10cond_traces_combined.extend(mpid_to_p_log10cond_traces_1e19[mpid])
#             log10cond_traces_combined.extend(mpid_to_n_log10cond_traces_1e19[mpid])
#             log10cond_traces_combined.extend(mpid_to_p_log10cond_traces_1e20[mpid])
#             log10cond_traces_combined.extend(mpid_to_n_log10cond_traces_1e20[mpid])
#
#             log10pf_traces_combined = list(mpid_to_p_log10pf_traces_1e16[mpid])
#             log10pf_traces_combined.extend(mpid_to_n_log10pf_traces_1e16[mpid])
#             log10pf_traces_combined.extend(mpid_to_p_log10pf_traces_1e17[mpid])
#             log10pf_traces_combined.extend(mpid_to_n_log10pf_traces_1e17[mpid])
#             log10pf_traces_combined.extend(mpid_to_p_log10pf_traces_1e18[mpid])
#             log10pf_traces_combined.extend(mpid_to_n_log10pf_traces_1e18[mpid])
#             log10pf_traces_combined.extend(mpid_to_p_log10pf_traces_1e19[mpid])
#             log10pf_traces_combined.extend(mpid_to_n_log10pf_traces_1e19[mpid])
#             log10pf_traces_combined.extend(mpid_to_p_log10pf_traces_1e20[mpid])
#             log10pf_traces_combined.extend(mpid_to_n_log10pf_traces_1e20[mpid])
#
#             log10S2_traces_combined = list(mpid_to_p_log10S2_traces_1e16[mpid])
#             log10S2_traces_combined.extend(mpid_to_n_log10S2_traces_1e16[mpid])
#             log10S2_traces_combined.extend(mpid_to_p_log10S2_traces_1e17[mpid])
#             log10S2_traces_combined.extend(mpid_to_n_log10S2_traces_1e17[mpid])
#             log10S2_traces_combined.extend(mpid_to_p_log10S2_traces_1e18[mpid])
#             log10S2_traces_combined.extend(mpid_to_n_log10S2_traces_1e18[mpid])
#             log10S2_traces_combined.extend(mpid_to_p_log10S2_traces_1e19[mpid])
#             log10S2_traces_combined.extend(mpid_to_n_log10S2_traces_1e19[mpid])
#             log10S2_traces_combined.extend(mpid_to_p_log10S2_traces_1e20[mpid])
#             log10S2_traces_combined.extend(mpid_to_n_log10S2_traces_1e20[mpid])
#
#             if formula not in formula_to_ener_per_atom or ener_per_atom < formula_to_ener_per_atom[formula]:
#                 # either the formula hasn't been seen yet,
#                 #  or we have a lower energy polymorph and we'll use this one instead
#                 formula_to_ener_per_atom[formula] = ener_per_atom
#                 formula_to_seebecks[formula] = seebeck_traces_combined
#                 formula_to_log10conds[formula] = log10cond_traces_combined
#                 formula_to_log10pfs[formula] = log10pf_traces_combined
#                 formula_to_log10S2s[formula] = log10S2_traces_combined
#                 formula_to_gap[formula] = mpid_to_gaps[mpid]
#                 formula_to_mpid[formula] = mpid
#                 formula_to_comp[formula] = composition
#                 formula_to_struct[formula] = mpid_to_struct[mpid]
#
#     skipatom_sum_seebeck_dataset = []
#     skipatom_sum_gap_seebeck_dataset = []
#     skipatom_sum_log10cond_dataset = []
#     skipatom_sum_gap_log10cond_dataset = []
#     fren_transformer_seebeck_dataset = []
#     fren_transformer_gap_seebeck_dataset = []
#     fren_transformer_log10cond_dataset = []
#     fren_transformer_gap_log10cond_dataset = []
#     meredig_seebeck_dataset = []
#     meredig_gap_seebeck_dataset = []
#     meredig_log10cond_dataset = []
#     meredig_gap_log10cond_dataset = []
#     fren_transformer_gap_struct_seebeck_dataset = []
#     fren_transformer_gap_struct_log10cond_dataset = []
#     meredig_gap_struct_seebeck_dataset = []
#     meredig_gap_struct_log10cond_dataset = []
#     fren_transformer_log10pf_dataset = []
#     fren_transformer_gap_log10pf_dataset = []
#     fren_transformer_gap_struct_log10pf_dataset = []
#     meredig_log10pf_dataset = []
#     meredig_gap_log10pf_dataset = []
#     meredig_gap_struct_log10pf_dataset = []
#     fren_transformer_log10S2_dataset = []
#     fren_transformer_gap_log10S2_dataset = []
#     meredig_log10S2_dataset = []
#     meredig_gap_log10S2_dataset = []
#     metadata = []
#     for formula in tqdm(formula_to_seebecks):
#         composition = formula_to_comp[formula]
#         seebeck_traces = formula_to_seebecks[formula]
#         log10cond_traces = formula_to_log10conds[formula]
#         log10pf_traces = formula_to_log10pfs[formula]
#         log10S2_traces = formula_to_log10S2s[formula]
#         gap = formula_to_gap[formula]
#         mpid = formula_to_mpid[formula]
#         struct = formula_to_struct[formula]
#         struct_vec = get_struct_vec(struct)
#
#         skipatom_sum_seebeck_dataset.append([
#             get_sum_pooled_vector(composition, skipatom_dictionary, skipatom_vectors), seebeck_traces
#         ])
#         skipatom_sum_gap_seebeck_dataset.append([
#             get_sum_pooled_vector(composition, skipatom_dictionary, skipatom_vectors, gap=gap), seebeck_traces
#         ])
#
#         skipatom_sum_log10cond_dataset.append([
#             get_sum_pooled_vector(composition, skipatom_dictionary, skipatom_vectors), log10cond_traces
#         ])
#         skipatom_sum_gap_log10cond_dataset.append([
#             get_sum_pooled_vector(composition, skipatom_dictionary, skipatom_vectors, gap=gap), log10cond_traces
#         ])
#
#         fren_transformer_seebeck_dataset.append([
#             get_fren_transformer_input(composition, skipatom_dictionary, skipatom_vectors), seebeck_traces
#         ])
#         fren_transformer_gap_seebeck_dataset.append([
#             get_fren_transformer_input(composition, skipatom_dictionary, skipatom_vectors, gap=gap), seebeck_traces
#         ])
#         fren_transformer_log10cond_dataset.append([
#             get_fren_transformer_input(composition, skipatom_dictionary, skipatom_vectors), log10cond_traces
#         ])
#         fren_transformer_gap_log10cond_dataset.append([
#             get_fren_transformer_input(composition, skipatom_dictionary, skipatom_vectors, gap=gap), log10cond_traces
#         ])
#
#         fren_transformer_gap_struct_seebeck_dataset.append([
#             get_fren_transformer_input(composition, skipatom_dictionary, skipatom_vectors, gap=gap, struct=struct_vec),
#             seebeck_traces
#         ])
#         fren_transformer_gap_struct_log10cond_dataset.append([
#             get_fren_transformer_input(composition, skipatom_dictionary, skipatom_vectors, gap=gap, struct=struct_vec),
#             log10cond_traces
#         ])
#         fren_transformer_log10pf_dataset.append([
#             get_fren_transformer_input(composition, skipatom_dictionary, skipatom_vectors), log10pf_traces
#         ])
#         fren_transformer_gap_log10pf_dataset.append([
#             get_fren_transformer_input(composition, skipatom_dictionary, skipatom_vectors, gap=gap), log10pf_traces
#         ])
#         fren_transformer_gap_struct_log10pf_dataset.append([
#             get_fren_transformer_input(composition, skipatom_dictionary, skipatom_vectors, gap=gap, struct=struct_vec),
#             log10pf_traces
#         ])
#
#         fren_transformer_log10S2_dataset.append([
#             get_fren_transformer_input(composition, skipatom_dictionary, skipatom_vectors), log10S2_traces
#         ])
#         fren_transformer_gap_log10S2_dataset.append([
#             get_fren_transformer_input(composition, skipatom_dictionary, skipatom_vectors, gap=gap), log10S2_traces
#         ])
#
#         meredig_vec = meredig.featurize(composition)  # it's not cheap to create the Meredig vector
#         # drop columns 108 and 109 (apparently range and mean AtomicRadius), which contain NaNs in some records
#         meredig_vec = [i for j, i in enumerate(meredig_vec) if j not in [108, 109]]
#         meredig_vec_with_gap = list(meredig_vec) + [gap]
#         meredig_vec_with_gap_struct = list(meredig_vec_with_gap) + struct_vec
#         meredig_seebeck_dataset.append([meredig_vec, seebeck_traces])
#         meredig_gap_seebeck_dataset.append([meredig_vec_with_gap, seebeck_traces])
#         meredig_log10cond_dataset.append([meredig_vec, log10cond_traces])
#         meredig_gap_log10cond_dataset.append([meredig_vec_with_gap, log10cond_traces])
#         meredig_gap_struct_seebeck_dataset.append([meredig_vec_with_gap_struct, seebeck_traces])
#         meredig_gap_struct_log10cond_dataset.append([meredig_vec_with_gap_struct, log10cond_traces])
#         meredig_log10pf_dataset.append([meredig_vec, log10pf_traces])
#         meredig_gap_log10pf_dataset.append([meredig_vec_with_gap, log10pf_traces])
#         meredig_gap_struct_log10pf_dataset.append([meredig_vec_with_gap_struct, log10pf_traces])
#         meredig_log10S2_dataset.append([meredig_vec, log10S2_traces])
#         meredig_gap_log10S2_dataset.append([meredig_vec_with_gap, log10S2_traces])
#
#         metadata.append((formula, mpid))
#
#     print(len(skipatom_sum_seebeck_dataset))
#     print(len(skipatom_sum_gap_seebeck_dataset))
#     print(len(skipatom_sum_log10cond_dataset))
#     print(len(skipatom_sum_gap_log10cond_dataset))
#     print(len(fren_transformer_seebeck_dataset))
#     print(len(fren_transformer_gap_seebeck_dataset))
#     print(len(fren_transformer_log10cond_dataset))
#     print(len(fren_transformer_gap_log10cond_dataset))
#     print(len(meredig_seebeck_dataset))
#     print(len(meredig_gap_seebeck_dataset))
#     print(len(meredig_log10cond_dataset))
#     print(len(meredig_gap_log10cond_dataset))
#     print(len(fren_transformer_gap_struct_seebeck_dataset))
#     print(len(fren_transformer_gap_struct_log10cond_dataset))
#     print(len(meredig_gap_struct_seebeck_dataset))
#     print(len(meredig_gap_struct_log10cond_dataset))
#     print(len(fren_transformer_log10pf_dataset))
#     print(len(fren_transformer_gap_log10pf_dataset))
#     print(len(fren_transformer_gap_struct_log10pf_dataset))
#     print(len(meredig_log10pf_dataset))
#     print(len(meredig_gap_log10pf_dataset))
#     print(len(meredig_gap_struct_log10pf_dataset))
#     print(len(fren_transformer_log10S2_dataset))
#     print(len(fren_transformer_gap_log10S2_dataset))
#     print(len(meredig_log10S2_dataset))
#     print(len(meredig_gap_log10S2_dataset))
#     print(len(metadata))
#
#     with gzip.open(os.path.join(data_out_dir, skipatom_sum_seebeck_file), "wb") as f:
#         pickle.dump((metadata, skipatom_sum_seebeck_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, skipatom_sum_gap_seebeck_file), "wb") as f:
#         pickle.dump((metadata, skipatom_sum_gap_seebeck_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, skipatom_sum_log10cond_file), "wb") as f:
#         pickle.dump((metadata, skipatom_sum_log10cond_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, skipatom_sum_gap_log10cond_file), "wb") as f:
#         pickle.dump((metadata, skipatom_sum_gap_log10cond_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, fren_transformer_seebeck_file), "wb") as f:
#         pickle.dump((metadata, fren_transformer_seebeck_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, fren_transformer_gap_seebeck_file), "wb") as f:
#         pickle.dump((metadata, fren_transformer_gap_seebeck_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, fren_transformer_log10cond_file), "wb") as f:
#         pickle.dump((metadata, fren_transformer_log10cond_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, fren_transformer_gap_log10cond_file), "wb") as f:
#         pickle.dump((metadata, fren_transformer_gap_log10cond_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, meredig_seebeck_file), "wb") as f:
#         pickle.dump((metadata, meredig_seebeck_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, meredig_gap_seebeck_file), "wb") as f:
#         pickle.dump((metadata, meredig_gap_seebeck_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, meredig_log10cond_file), "wb") as f:
#         pickle.dump((metadata, meredig_log10cond_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, meredig_gap_log10cond_file), "wb") as f:
#         pickle.dump((metadata, meredig_gap_log10cond_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, fren_transformer_gap_struct_seebeck_file), "wb") as f:
#         pickle.dump((metadata, fren_transformer_gap_struct_seebeck_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, fren_transformer_gap_struct_log10cond_file), "wb") as f:
#         pickle.dump((metadata, fren_transformer_gap_struct_log10cond_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, meredig_gap_struct_seebeck_file), "wb") as f:
#         pickle.dump((metadata, meredig_gap_struct_seebeck_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, meredig_gap_struct_log10cond_file), "wb") as f:
#         pickle.dump((metadata, meredig_gap_struct_log10cond_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, fren_transformer_log10pf_file), "wb") as f:
#         pickle.dump((metadata, fren_transformer_log10pf_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, fren_transformer_gap_log10pf_file), "wb") as f:
#         pickle.dump((metadata, fren_transformer_gap_log10pf_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, fren_transformer_gap_struct_log10pf_file), "wb") as f:
#         pickle.dump((metadata, fren_transformer_gap_struct_log10pf_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, meredig_log10pf_file), "wb") as f:
#         pickle.dump((metadata, meredig_log10pf_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, meredig_gap_log10pf_file), "wb") as f:
#         pickle.dump((metadata, meredig_gap_log10pf_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, meredig_gap_struct_log10pf_file), "wb") as f:
#         pickle.dump((metadata, meredig_gap_struct_log10pf_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, fren_transformer_log10S2_file), "wb") as f:
#         pickle.dump((metadata, fren_transformer_log10S2_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, fren_transformer_gap_log10S2_file), "wb") as f:
#         pickle.dump((metadata, fren_transformer_gap_log10S2_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, meredig_log10S2_file), "wb") as f:
#         pickle.dump((metadata, meredig_log10S2_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
#
#     with gzip.open(os.path.join(data_out_dir, meredig_gap_log10S2_file), "wb") as f:
#         pickle.dump((metadata, meredig_gap_log10S2_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)
