from os import listdir
from os.path import isfile, join
import argparse
import json
import gzip
import csv

GGA = "GGA"
GGA_U = "GGA+U"


def extract_xyz(tensor):
    return tensor[0][0], tensor[1][1], tensor[2][2]


def extract_from_functional(data, functional, mp_id, writer):
    for doping_type in ["p", "n"]:
        for temperature in ["100", "200", "300", "400", "500", "600", "700", "800", "900", "1000", "1100", "1200", "1300"]:
            for doping_level in ["1e+16", "1e+17", "1e+18", "1e+19", "1e+20"]:
                seebeck_xx, seebeck_yy, seebeck_zz = extract_xyz(
                    data["seebeck_doping"][doping_type][temperature][doping_level]["tensor"]
                )
                cond_xx, cond_yy, cond_zz = extract_xyz(
                    data["cond_doping"][doping_type][temperature][doping_level]["tensor"]
                )
                kappa_xx, kappa_yy, kappa_zz = extract_xyz(
                    data["kappa_doping"][doping_type][temperature][doping_level]["tensor"]
                )

                writer.writerow([mp_id, functional, doping_type, temperature, doping_level,
                                 seebeck_xx, seebeck_yy, seebeck_zz, cond_xx, cond_yy, cond_zz,
                                 kappa_xx, kappa_yy, kappa_zz])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', action='store',
                        default=None, nargs='+',
                        help='The directories containing the .json.gz files to process.')
    parser.add_argument('--out', '-o', action='store',
                        default=None,
                        help='The path to the .csv file to be created.')
    args = parser.parse_args()

    with open(args.out, "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["mp_id", "functional", "doping_type", "temperature", "doping_level",
                             "seebeck_xx", "seebeck_yy", "seebeck_zz", "cond_xx", "cond_yy", "cond_zz",
                             "kappa_xx", "kappa_yy", "kappa_zz"])

        for dir_arg in args.dir:
            files = [f for f in listdir(dir_arg) if isfile(join(dir_arg, f))]
            for file in files:
                print("processing %s" % join(dir_arg, file))
                with gzip.open(join(dir_arg, file), "rt") as f:
                    res = json.load(f)
                    mpid = res["mp_id"]
                    if GGA in res:
                        extract_from_functional(res[GGA], GGA, mpid, csv_writer)
                    if GGA_U in res:
                        extract_from_functional(res[GGA_U], GGA_U, mpid, csv_writer)
                    if not GGA in res and not GGA_U in res:
                        raise Exception("could not find GGA or GGA+U data in %s" % join(dir_arg, file))
