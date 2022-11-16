import sys
import os
import argparse
from tqdm import tqdm
import requests


STORAGE_URL = "https://cratenet-data.s3.amazonaws.com"
RICCI_DATA_XYZ = f"{STORAGE_URL}/ricci_data_xyz.csv.gz"
RICCI_DATA_GAP = f"{STORAGE_URL}/ricci_data_gap.csv.gz"
SEEBECK_MPID_TRACES = f"{STORAGE_URL}/seebeck_mpid_traces.csv.gz"
SEEBECK_COMP_TRACES = f"{STORAGE_URL}/seebeck_comp_traces.csv.gz"
COND_MPID_TRACES = f"{STORAGE_URL}/cond_mpid_traces.csv.gz"
COND_COMP_TRACES = f"{STORAGE_URL}/cond_comp_traces.csv.gz"
PF_MPID_TRACES = f"{STORAGE_URL}/pf_mpid_traces.csv.gz"
PF_COMP_TRACES = f"{STORAGE_URL}/pf_comp_traces.csv.gz"
COMP_GAPS = f"{STORAGE_URL}/comp_gaps.csv"
COMP_TO_MPID = f"{STORAGE_URL}/comp_to_mpid.csv"
RF_SEEBECK_DATASET = f"{STORAGE_URL}/rf_seebeck_dataset.pkl.gz"
RF_LOG10COND_DATASET = f"{STORAGE_URL}/rf_log10cond_dataset.pkl.gz"
RF_LOG10PF_DATASET = f"{STORAGE_URL}/rf_log10pf_dataset.pkl.gz"
RF_SEEBECK_GAP_DATASET = f"{STORAGE_URL}/rf_seebeck_gap_dataset.pkl.gz"
RF_LOG10COND_GAP_DATASET = f"{STORAGE_URL}/rf_log10cond_gap_dataset.pkl.gz"
RF_LOG10PF_GAP_DATASET = f"{STORAGE_URL}/rf_log10pf_gap_dataset.pkl.gz"

BLOCK_SIZE = 1024


def get_url(option):
    if option == "gap":
        return RICCI_DATA_GAP
    elif option == "xyz":
        return RICCI_DATA_XYZ
    elif option == "seebeck_mpid_traces":
        return SEEBECK_MPID_TRACES
    elif option == "seebeck_comp_traces":
        return SEEBECK_COMP_TRACES
    elif option == "cond_mpid_traces":
        return COND_MPID_TRACES
    elif option == "cond_comp_traces":
        return COND_COMP_TRACES
    elif option == "pf_mpid_traces":
        return PF_MPID_TRACES
    elif option == "pf_comp_traces":
        return PF_COMP_TRACES
    elif option == "comp_gaps":
        return COMP_GAPS
    elif option == "comp_to_mpid":
        return COMP_TO_MPID
    elif option == "rf_seebeck_dataset":
        return RF_SEEBECK_DATASET
    elif option == "rf_log10cond_dataset":
        return RF_LOG10COND_DATASET
    elif option == "rf_log10pf_dataset":
        return RF_LOG10PF_DATASET
    elif option == "rf_seebeck_gap_dataset":
        return RF_SEEBECK_GAP_DATASET
    elif option == "rf_log10cond_gap_dataset":
        return RF_LOG10COND_GAP_DATASET
    elif option == "rf_log10pf_gap_dataset":
        return RF_LOG10PF_GAP_DATASET
    else:
        raise Exception(f"unsupported option: {option}")


def get_out_path(out_dir, url):
    fname = url.split('/')[-1]

    if out_dir is None:
        return f"./{fname}"

    if not os.path.exists(out_dir):
        print(f"creating non-existent directory: {out_dir}")
        os.makedirs(out_dir)

    return os.path.join(out_dir, fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    type_choices = ["gap", "xyz", "seebeck_mpid_traces", "cond_mpid_traces", "pf_mpid_traces", "seebeck_comp_traces",
                    "cond_comp_traces", "pf_comp_traces", "comp_gaps", "comp_to_mpid", "rf_seebeck_dataset",
                    "rf_log10cond_dataset", "rf_log10pf_dataset",  "rf_seebeck_gap_dataset", "rf_log10cond_gap_dataset",
                    "rf_log10pf_gap_dataset"]
    parser.add_argument("type", choices=type_choices,
                        help=f"Supported options: {', '.join(type_choices)}")
    parser.add_argument("--out", "-o", action="store",
                        required=False,
                        help="The path to the local directory where the downloaded file will be stored. "
                             "If the directory does not exist, it will be created.")
    args = parser.parse_args()

    url = get_url(args.type)
    out_path = get_out_path(args.out, url)

    print(f"downloading to {out_path} ...")

    response = requests.get(url, stream=True)

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(out_path, "wb") as f:
        for data in response.iter_content(BLOCK_SIZE):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("error downloading!")
        sys.exit(1)

    print("done!")
