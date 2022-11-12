import argparse
import csv
import gzip
import numpy as np


def get_seebeck_trace(csv_line):
    return np.mean([float(csv_line[5]), float(csv_line[6]), float(csv_line[7])])


def get_cond_trace(csv_line):
    return np.mean([float(csv_line[8]), float(csv_line[9]), float(csv_line[10])])


def get_pf_trace(csv_line):
    seebeck1 = float(csv_line[5])  # μV/K
    cond1 = float(csv_line[8])  # (Ω m s)^-1
    pf1 = (((seebeck1 / 1e6) ** 2) * (cond1 * 1e6)) / 100  # μW/(cm K^2 s)

    seebeck2 = float(csv_line[6])  # μV/K
    cond2 = float(csv_line[9])  # (Ω m s)^-1
    pf2 = (((seebeck2 / 1e6) ** 2) * (cond2 * 1e6)) / 100  # μW/(cm K^2 s)

    seebeck3 = float(csv_line[7])  # μV/K
    cond3 = float(csv_line[10])  # (Ω m s)^-1
    pf3 = (((seebeck3 / 1e6) ** 2) * (cond3 * 1e6)) / 100  # μW/(cm K^2 s)

    return np.mean([pf1, pf2, pf3])


def get_trace(prop):
    if prop == "seebeck":
        return get_seebeck_trace
    elif prop == "cond":
        return get_cond_trace
    elif prop == "pf":
        return get_pf_trace
    else:
        raise Exception(f"unrecognized property: {prop}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("prop", choices=["seebeck", "cond", "pf"], help="Supported options: seebeck, cond, pf")
    parser.add_argument("--data", nargs="?", required=True, type=str,
                        help="path to the extracted tensor diagonals data file (must be a .csv.gz file)")
    parser.add_argument("--out", nargs="?", required=True, type=str,
                        help="path to the output file; a .csv.gz extension should be used")
    args = parser.parse_args()

    all_data_xyz_file = args.data
    out_trace_file = args.out
    property = args.prop

    all_keys_to_trace = {}
    trace_fn = get_trace(property)

    print("processing records...")

    with gzip.open(all_data_xyz_file, "rt") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            mpid = line[0]
            functional = line[1]  # 'GGA' or 'GGA+U'
            doping_type = line[2]
            temperature = line[3]
            doping_level = line[4]
            trace = trace_fn(line)

            key = (mpid, doping_type, doping_level)

            if key not in all_keys_to_trace:
                temperature_to_trace_and_functional = {str(x): [np.nan, ""] for x in list(range(100, 1400, 100))}
                all_keys_to_trace[key] = temperature_to_trace_and_functional

            existing_trace = all_keys_to_trace[key][temperature][0]
            existing_functional = all_keys_to_trace[key][temperature][1]

            if not np.isnan(existing_trace):
                print(f"found existing trace for {key[0]} {key[1]} {key[2]} at {temperature}K: {existing_trace} "
                      f"({existing_functional}); incoming: {trace} ({functional})")
                if existing_functional == "GGA+U":
                    print("existing is GGA+U; skipping")
                    continue
            all_keys_to_trace[key][temperature][0] = trace
            all_keys_to_trace[key][temperature][1] = functional

    print(f"writing {out_trace_file} file...")

    with gzip.open(out_trace_file, "wt") as f:
        writer = csv.writer(f)
        for key in all_keys_to_trace:
            mpid, doping_type, doping_level = key
            writer.writerow([
                mpid,                              # e.g. mp-716509
                doping_type,                       # e.g. p
                doping_level,                      # e.g. 1e16
                all_keys_to_trace[key]["100"][0],  # trace @ 100K (e.g. 507.47166)
                all_keys_to_trace[key]["200"][0],  # trace @ 200K (e.g. 507.47166)
                all_keys_to_trace[key]["300"][0],  # trace @ 300K (e.g. 507.47166)
                all_keys_to_trace[key]["400"][0],  # trace @ 400K (e.g. 507.47166)
                all_keys_to_trace[key]["500"][0],  # trace @ 500K (e.g. 507.47166)
                all_keys_to_trace[key]["600"][0],  # trace @ 600K (e.g. 507.47166)
                all_keys_to_trace[key]["700"][0],  # trace @ 700K (e.g. 507.47166)
                all_keys_to_trace[key]["800"][0],  # trace @ 800K (e.g. 507.47166)
                all_keys_to_trace[key]["900"][0],  # trace @ 900K (e.g. 507.47166)
                all_keys_to_trace[key]["1000"][0], # trace @ 1000K (e.g. 507.47166)
                all_keys_to_trace[key]["1100"][0], # trace @ 1100K (e.g. 507.47166)
                all_keys_to_trace[key]["1200"][0], # trace @ 1200K (e.g. 507.47166)
                all_keys_to_trace[key]["1300"][0]  # trace @ 1300K (e.g. 507.47166)
            ])

    print("done!")
