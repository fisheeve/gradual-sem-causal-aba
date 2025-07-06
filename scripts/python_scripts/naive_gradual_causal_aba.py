import sys  # noqa
sys.path.insert(0, 'ArgCausalDisco/')  # noqa
sys.path.insert(0, 'notears/')  # noqa
sys.path.insert(0, 'GradualABA/')  # noqa


import time
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from src.utils.configure_r import configure_r
from src.abapc import get_cg_and_facts
from src.utils.bn_utils import get_dataset
from src.gradual.naive.causal_bsaf import NaiveCausalBSAF
from src.gradual.model_wrappers import ModelWrapper, ModelEnum
from ArgCausalDisco.utils.helpers import random_stability, logger_setup
from ArgCausalDisco.utils.graph_utils import DAGMetrics, dag2cpdag
from ArgCausalDisco.utils.data_utils import simulate_dag
from ArgCausalDisco.cd_algorithms.models import run_method


ALPHA = 0.01
INDEP_TEST = 'fisherz'
EDGE_NODE_RATIO = 1
DATASETS = [
    'cancer',
    'earthquake',
    'survey',
    'asia'
]


def parse_args():
    parser = ArgumentParser(description="Compare ABAPC implementations on random DAGs")
    parser.add_argument('--n-runs', type=int, default=50, help='Number of runs for the experiment')
    parser.add_argument('--sample-size', type=int, default=5000, help='Sample size for the random DAGs')
    parser.add_argument('--output-dir', type=str, default='results_pure_aba', help='Directory to save results')
    return parser.parse_args()


def main():
    args = parse_args()
    configure_r()
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger_setup(str(out_path / 'log_compare_sem_bnlearn.log'))

    random_stability(2024)
    seeds_list = np.random.randint(0, 10000, (args.n_runs,)).tolist()
    df = pd.DataFrame()

    for dataset_name in tqdm(DATASETS, desc="tqdm Datasets"):

        for seed in tqdm(seeds_list, desc="tqdm Seeds"):
            X_s, B_true = get_dataset(dataset_name,
                                      seed=seed,
                                      sample_size=args.sample_size)

            for method in ['random', 'mpc', 'naive_gradual_causal_aba']:
                # random method

                random_stability(seed)
                metadata = None
                if method == 'random':
                    start = time.time()
                    W_est = simulate_dag(d=B_true.shape[1], s0=B_true.sum().astype(int), graph_type='ER')
                    elapsed = time.time() - start
                elif method == 'mpc':
                    W_est, elapsed = run_method(X_s,
                                                method,
                                                seed,
                                                test_alpha=ALPHA,
                                                test_name=INDEP_TEST,
                                                device='0',
                                                scenario=f"{method}_{dataset_name}")

                elif method == 'naive_gradual_causal_aba':
                    start = time.time()
                    _, facts = get_cg_and_facts(X_s,
                                                alpha=ALPHA,
                                                indep_test=INDEP_TEST,
                                                uc_rule=5,
                                                stable=True)
                    causal_bsaf_wrapper = NaiveCausalBSAF(n_nodes=X_s.shape[1],
                                                          facts=facts)
                    causal_bsaf = causal_bsaf_wrapper.create_bsaf()
                    model = ModelWrapper(causal_bsaf,
                                         n_nodes=X_s.shape[1],
                                         model_name=ModelEnum.DF_QUAD)
                    model.solve()
                    W_est = model.build_greedy_graph()
                    elapsed = time.time() - start
                    metadata = json.dumps(model.model.graph_data)

                B_est = (W_est != 0).astype(int)
                mt_cpdag = DAGMetrics(dag2cpdag(B_est), B_true).metrics
                B_est = (W_est > 0).astype(int)
                mt_dag = DAGMetrics(B_est, B_true).metrics

                df = pd.concat([df, pd.DataFrame([{
                    'method': method,
                    'elapsed': elapsed,
                    'n_nodes': X_s.shape[1],
                    'dataset_name': dataset_name,
                    'seed': seed,
                    'best_model': json.dumps(B_est.tolist()),
                    'metadata': metadata,
                    **{'mt_dag_'+k: v for k, v in mt_dag.items()},
                    **{'mt_cpdag_'+k: v for k, v in mt_cpdag.items()}
                }])])
                df.to_csv(out_path / f'naive_gradual_causal_aba.csv', index=False)


if __name__ == "__main__":
    main()
