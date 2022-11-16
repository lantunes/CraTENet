import argparse
import csv
import gzip
from tqdm import tqdm


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


def get_mpid_to_traces(traces_file):
    mpid_to_traces = {}
    with gzip.open(traces_file, "rt") as f:
        reader = csv.reader(f)
        for line in reader:
            key = (line[0], line[1], line[2])
            vals = [line[i] for i in range(3, 16)]
            mpid_to_traces[key] = vals
    return mpid_to_traces


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs=2, required=False, type=str, action="append",
                        help="the first arg is a path to the extracted tensor diagonals data file "
                             "(must be a .csv.gz file); the second arg is the path the destination file containing the "
                             "dedeplicated entries (should contain a .csv.gz extension)")
    parser.add_argument("--formulas", nargs=1, required=True, type=str,
                        help="path to the .csv file containing a mapping from MP ID to composition")
    parser.add_argument("--energies", nargs=1, required=True, type=str,
                        help="path to the .csv.gz file containing a mapping from MP ID to energy per atom")
    parser.add_argument("--gaps", nargs=2, required=True, type=str,
                        help="the first arg is the path the .csv file containing a mapping from MP IDs to gaps; the "
                             "second arg is a path to the deduplicated .csv gaps file that will be written, containing "
                             "a mapping from compositions to gaps")
    parser.add_argument("--mpids", nargs="?", required=False, type=str,
                        help="path to the .csv file containing a mapping from compositions to the MP ID they represent,"
                             " after disambiguating the duplicates")
    args = parser.parse_args()

    all_gaps_file = args.gaps[0]
    target_gaps_file = args.gaps[1]
    mp_all_ener_per_atom_file = args.energies[0]
    all_formulas_file = args.formulas[0]
    mpids_file = args.mpids
    traces = args.traces

    mpid_to_gaps = get_all_mpid_to_gaps(all_gaps_file)
    mpid_to_ener_per_atom = get_all_mpid_to_ener_per_atom(mp_all_ener_per_atom_file)

    print("determining lowest energy polymorph for each composition...")

    formula_to_ener_per_atom = {}
    formula_to_gap = {}
    formula_to_mpid = {}
    with open(all_formulas_file, "rt") as in_f:
        reader = csv.reader(in_f)
        for line in tqdm(reader):
            mpid = line[0]
            formula = line[1]

            ener_per_atom = mpid_to_ener_per_atom[mpid]

            if formula not in formula_to_ener_per_atom or ener_per_atom < formula_to_ener_per_atom[formula]:
                # either the formula hasn't been seen yet,
                #  or we have a lower energy polymorph and we'll use this one instead
                formula_to_ener_per_atom[formula] = ener_per_atom
                formula_to_gap[formula] = mpid_to_gaps[mpid]
                formula_to_mpid[formula] = mpid

    print(f"number of unique formulas: {len(formula_to_mpid):,}")

    if traces is not None:
        for mpid_traces_file, comp_traces_file in traces:
            mpid_traces = get_mpid_to_traces(mpid_traces_file)
            print(f"writing {comp_traces_file}...")
            with gzip.open(comp_traces_file, "wt") as f:
                writer = csv.writer(f)
                n_entries = 0
                for formula, mpid in tqdm(formula_to_mpid.items()):
                    for doping_type in ["p", "n"]:
                        for doping_level in ["1e+16", "1e+17", "1e+18", "1e+19", "1e+20"]:
                            entry = [formula, doping_type, doping_level]
                            entry.extend(mpid_traces[(mpid, doping_type, doping_level)])
                            writer.writerow(entry)
                            n_entries += 1
                print(f"number of entries written: {n_entries:,}")

    print(f"writing {target_gaps_file}...")

    with open(target_gaps_file, "wt") as f:
        writer = csv.writer(f)
        n_entries = 0
        for formula, gap in tqdm(formula_to_gap.items()):
            writer.writerow([formula, gap])
            n_entries += 1

    print(f"number of entries written: {n_entries:,}")

    if mpids_file is not None:
        print(f"writing {mpids_file}...")
        with open(mpids_file, "wt") as f:
            writer = csv.writer(f)
            for formula, mpid in formula_to_mpid.items():
                writer.writerow([formula, mpid])
        print(f"number of entries written: {len(formula_to_mpid):,}")
