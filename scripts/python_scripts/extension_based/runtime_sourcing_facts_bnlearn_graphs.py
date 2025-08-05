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
from ArgCausalDisco.utils.helpers import random_stability
from src.utils.bn_utils import get_dataset


SEED = 2024
ALPHA = 0.01
INDEP_TEST = 'fisherz'
SAMPLE_SIZE = 5000

RESULT_DIR = Path("./results/existing/runtime_sourcing_facts_bnlearn_graphs")
DATASETS = ['cancer', 'earthquake', 'survey', 'asia', 'sachs', 'child', 'insurance']

DAG_NODES_MAP = {'cancer': 5,
                 'earthquake': 5,
                 'survey': 6,
                 'asia': 8,
                 'sachs': 11,
                 'child': 20,
                 'insurance': 27}

def parse_args():
    parser = argparse.ArgumentParser(description="runtime_sourcing_facts_bnlearn_graphs")
    parser.add_argument('--n-runs', type=int, default=50, help='Number of runs for each dataset')
    return parser.parse_args()

def main(n_runs):
    results = pd.DataFrame()

    for dataset in DATASETS:
        n_nodes = DAG_NODES_MAP[dataset]
        logger.info(f"Running experiment for dataset: {dataset}")

        random_stability(SEED)
        seeds_list = np.random.randint(0, 10000, (n_runs,)).tolist()

        for seed in tqdm(seeds_list, desc=f"Running dataset {dataset} with {n_runs} seeds"):
            X_s, _ = get_dataset(dataset,
                                 seed=seed,
                                 sample_size=SAMPLE_SIZE)
            random_stability(seed)
            start = time.time()
            cg, facts = get_cg_and_facts(X_s, alpha=ALPHA, indep_test=INDEP_TEST)
            elapsed = time.time() - start

            results = pd.concat([results, pd.DataFrame({
                'n_nodes': [n_nodes],
                'dataset': [dataset],
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
