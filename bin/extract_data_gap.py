from os import listdir
from os.path import isfile, join
import argparse
import json
import gzip
import csv

GGA = "GGA"
GGA_U = "GGA+U"


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
        csv_writer.writerow(["mp_id", "functional", "gap"])

        for dir_arg in args.dir:
            files = [f for f in listdir(dir_arg) if isfile(join(dir_arg, f))]
            for file in files:
                print("processing %s" % join(dir_arg, file))
                with gzip.open(join(dir_arg, file), "rt") as f:
                    res = json.load(f)
                    mpid = res["mp_id"]
                    gap = res["gap"]
                    if GGA in gap:
                        csv_writer.writerow([mpid, GGA, gap[GGA]])
                    if GGA_U in gap:
                        csv_writer.writerow([mpid, GGA_U, gap[GGA_U]])
                    if not GGA in gap and not GGA_U in gap:
                        raise Exception("could not find GGA or GGA+U data in %s" % join(dir_arg, file))
