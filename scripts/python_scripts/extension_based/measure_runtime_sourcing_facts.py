import sys
sys.path.insert(0, 'GradualABA/')
sys.path.insert(0, 'ArgCausalDisco/')

import time
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from logger import logger

from src.abapc import get_cg_and_facts
from src.utils.gen_random_nx import generate_random_bn_data
from ArgCausalDisco.utils.helpers import random_stability


SEED = 2024
ALPHA = 0.01
INDEP_TEST = 'fisherz'
SAMPLE_SIZE = 5000

RESULT_DIR = Path("./results/existing/fact_sourcing_runtime")
N_NODES = [3, 4, 5, 6, 7, 8, 10, 15, 20]
N_NODES_DOUBLE = [10, 15, 20]


def parse_args():
    parser = argparse.ArgumentParser(description="Run gradual ABA experiment v1")
    parser.add_argument('--n-runs', type=int, default=50, help='Number of runs for each dataset')
    return parser.parse_args()

def main(n_runs):
    results = pd.DataFrame()

    for n_nodes in N_NODES:
        if n_nodes in N_NODES_DOUBLE:
            n_edges_candidates = [n_nodes, n_nodes * 2]
        else:
            n_edges_candidates = [n_nodes]
        for n_edges in n_edges_candidates:
            logger.info(f"Running experiment for n_nodes: {n_nodes}, n_edges: {n_edges}")

            random_stability(SEED)
            seeds_list = np.random.randint(0, 10000, (n_runs,)).tolist()


            for seed in tqdm(seeds_list, desc=f"Running |V|={n_nodes} |E|={n_edges} with {n_runs} seeds"):
                X_s, _ = generate_random_bn_data(
                        n_nodes=n_nodes,
                        n_edges=n_edges,
                        n_samples=SAMPLE_SIZE,
                        seed=seed,
                        standardise=True
                )
                random_stability(seed)
                start = time.time()
                cg, facts = get_cg_and_facts(X_s, alpha=ALPHA, indep_test=INDEP_TEST)
                elapsed = time.time() - start

                results = pd.concat([results, pd.DataFrame({
                    'n_nodes': [n_nodes],
                    'n_edges': [n_edges],
                    'seed': [seed],
                    'elapsed_time': [elapsed]
                })], ignore_index=True)
                results.to_csv(RESULT_DIR / 'runtime_results.csv', index=False)



if __name__ == "__main__":
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    main(args.n_runs)
    print("Runtime sourcing facts experiment completed successfully!")
    print(f"Results saved to {RESULT_DIR / 'runtime_results.csv'}")
