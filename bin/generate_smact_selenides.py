import argparse
from smact import Element
from smact.screening import smact_filter
import itertools
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from pymatgen import Composition
from pymatgen.core import Element as PymatgenElement


def comp_maker(comp):
    form = []
    for el, ammt in zip(comp[0], comp[2]):
        form.append(el)
        form.append(ammt)
    form = ''.join(str(e) for e in form)
    pmg_form = Composition(form).reduced_formula
    return pmg_form


def listener(queue, fname, n):
    pbar = tqdm(total=n)
    seen = set()  # keep track of the formulas received so that we avoid duplicates
    with open(fname, "wt") as f:
        while True:
            message = queue.get()
            if message == "kill":
                break

            comps = [c for c in message]

            for comp in comps:
                formula = comp_maker(comp)
                if formula not in seen:
                    f.write("%s\n" % formula)
                seen.add(formula)

            pbar.update(1)


def worker(element_lists, queue):
    for i in range(len(element_lists)):
        try:
            filtered = smact_filter(element_lists[i])
            queue.put(filtered)
        except Exception as e:
            print("ERROR: %s" % e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", nargs="?", required=True, type=str,
                        help="path to the output file containing the generated compositions; "
                             "a .txt extension should be used")
    parser.add_argument("--max-z", nargs="?", required=False, type=int, default=83,
                        help="use only the elements with atomic number less than or equal to the given Z")
    parser.add_argument("--cations", nargs="?", required=False, type=int, default=2,
                        help="the number of cations to use")
    parser.add_argument("--processes", nargs="?", required=False, type=int, default=4,
                        help="the number of processes to utilize")
    parser.add_argument("--workers", nargs="?", required=False, type=int, default=1,
                        help="the number of workers to utilize")
    args = parser.parse_args()

    out_file = args.out
    max_z = args.max_z
    number_of_cations = args.cations
    processes = args.processes
    workers = args.workers

    selenium = "Se"
    elements = [elem.name for elem in sorted([e for e in PymatgenElement], key=lambda e: e.number)]

    # Define any elements to drop from the search
    more_electronegative = ["S", "N", "Cl", "O", "F"]
    nobel = ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]
    heavy = elements[max_z:]
    se = [selenium]
    disallowed = set(more_electronegative + nobel + se + heavy)

    allowed = [Element(e) for e in elements if e not in disallowed and Element(e).pauling_eneg]

    # Generate all combinations
    metal_pairs = itertools.combinations(allowed, number_of_cations)

    # Add Se to each combination
    systems = [[*m, Element(selenium)] for m in metal_pairs]

    chunks = np.array_split(np.array(systems), workers)

    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(processes=processes)

    watcher = pool.apply_async(listener, (queue, out_file, len(systems)))

    jobs = []
    for w in range(workers):
        chunk = chunks[w]
        job = pool.apply_async(worker, (chunk, queue))
        jobs.append(job)

    for job in jobs:
        job.get()

    queue.put("kill")
    pool.close()
    pool.join()
