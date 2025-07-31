'''
Experiment with V2 scalable implementation.
On random generated bayesian graphs.
'''

import sys
sys.path.insert(0, 'aspforaba/')
sys.path.insert(0, 'GradualABA/')
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')

import argparse
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tqdm import tqdm

from ArgCausalDisco.utils.helpers import random_stability
from GradualABA.semantics.modular.LinearInfluence import LinearInfluence
from GradualABA.semantics.modular.SetMinAggregation import SetMinAggregation
from logger import logger
from src.abapc import get_cg_and_facts, score_model_original
from src.gradual.run import reset_weights, run_model, set_weights_according_to_facts
from src.gradual.scalable.bsaf_builder_v2 import BSAFBuilderV2
from src.gradual.search_best_model import limited_depth_search_best_model
from src.gradual.semantic_modules.TopDiffAggregation import TopDiffAggregation
from src.utils.configure_r import configure_r
from src.utils.gen_random_nx import generate_random_bn_data
from src.utils.metrics import get_metrics
from src.utils.resource_utils import MemoryUsageExceededException, TimeoutException, timeout
from src.utils.utils import check_arrows_dag, get_matrix_from_arrow_set, parse_arrow


configure_r()

ALPHA = 0.01
INDEP_TEST = 'fisherz'

SAMPLE_SIZE = 5000
RESULT_DIR = Path("./results/gradual/v2_ablation_random_graphs")

N_NODES = [3, 4, 5, 6, 7, 8, 10, 15, 20]
N_NODES_DOUBLE = [10, 15, 20]

CYCLE_LENGTH = [3, 4, 5]
CT_DEPTH = [-1, 0, 1, 2, 3, 4]
PATH_LENGTH = [2, 3, 4]
C_SET_SIZE = [1, 2, 3]
SEARCH_DEPTH = [5, 10, 15]

SEED = 2024

TIMEOUT = 30 * 60  # 30 minutes


@dataclass
class ParamSet:
    cycle_length: int
    ct_depth: int
    path_length: int
    c_set_size: int
    search_depth: int


def is_param_set_valid(n_nodes,
                       param_set: ParamSet) -> bool:
    is_valid = (
        param_set.cycle_length <= n_nodes and
        param_set.ct_depth <= n_nodes-1 and
        param_set.path_length <= n_nodes-1 and
        param_set.c_set_size <= n_nodes-2
    )
    return is_valid


@timeout(TIMEOUT)  # Set a timeout of 30 minutes for the model run
def run_model_with_timeout(n_nodes, bsaf, model_name, set_aggregation, aggregation, influence, conservativeness, iterations):
    return run_model(
        n_nodes=n_nodes,
        bsaf=bsaf,
        model_name=model_name,
        set_aggregation=set_aggregation,
        aggregation=aggregation,
        influence=influence,
        conservativeness=conservativeness,
        iterations=iterations
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run gradual ABA experiment v2 on random graphs")
    parser.add_argument('--n-runs', type=int, default=50, help='Number of runs for each dataset')
    return parser.parse_args()


def main(n_runs, steps_ahead):
    param_sets = [
        ParamSet(cycle_length=cycle_length,
                 ct_depth=ct_depth,
                 path_length=path_length,
                 c_set_size=c_set_size,
                 search_depth=search_depth)
        for cycle_length in CYCLE_LENGTH
        for ct_depth in CT_DEPTH
        for path_length in PATH_LENGTH
        for c_set_size in C_SET_SIZE
        for search_depth in SEARCH_DEPTH
    ]

    facts_path = Path('./facts')
    facts_path.mkdir(parents=True, exist_ok=True)
    cpdag_metrics_df = pd.DataFrame()
    dag_metrics_df = pd.DataFrame()

    for n_nodes in N_NODES:
        n_edges_candidates = [n_nodes]
        if n_nodes in N_NODES_DOUBLE:
            n_edges_candidates.append(n_nodes * 2)  # double the number of nodes
        for n_edges in n_edges_candidates:
            for param_set in param_sets:
                if not is_param_set_valid(n_nodes, param_set):
                    continue
                dataset = f'random_graphs_n{n_nodes}_e{n_edges}'
                logger.info(f"Running experiment for dataset: {dataset}, param_set: {param_set}")

                random_stability(SEED)
                seeds_list = np.random.randint(0, 10000, (n_runs,)).tolist()
                n_nodes = N_NODES[dataset]

                # create bsaf
                start_bsaf_creation = time.time()
                try:
                    bsaf_builder = BSAFBuilderV2(
                        n_nodes=n_nodes,
                        max_cycle_size=n_nodes,
                        max_collider_tree_depth=n_nodes-1,
                        max_path_length=n_nodes-1,
                        max_conditioning_set_size=n_nodes-2)  # everything is maximal for given n_nodes, full scale
                    bsaf = bsaf_builder.create_bsaf()
                    assumptions_dict = bsaf_builder.name_to_assumption
                except MemoryUsageExceededException as e:
                    logger.error(
                        f"Memory usage exceeded while creating BSAF for {dataset} with param_set {param_set}: {e}")
                    continue
                elapsed_bsaf_creation = time.time() - start_bsaf_creation

                for seed in tqdm(seeds_list, desc=f"Running {dataset} with {n_runs} seeds"):
                    X_s, B_true = generate_random_bn_data(
                        n_nodes=n_nodes,
                        n_edges=n_edges,
                        n_samples=SAMPLE_SIZE,
                        seed=seed,
                        standardise=True
                    )
                    random_stability(seed)
                    cg, facts = get_cg_and_facts(X_s, alpha=ALPHA, indep_test=INDEP_TEST)
                    facts = sorted(facts, key=lambda x: x.score, reverse=True)  # sort by score

                    # run gradual aba to get the strengths
                    start_model_solution = time.time()
                    reset_weights(assumptions_dict)
                    set_weights_according_to_facts(assumptions_dict, facts)
                    try:
                        output = run_model_with_timeout(
                            n_nodes=n_nodes,
                            bsaf=bsaf,
                            model_name=None,
                            set_aggregation=SetMinAggregation(),
                            aggregation=TopDiffAggregation(),
                            influence=LinearInfluence(conservativeness=1),
                            conservativeness=1,
                            iterations=25,
                        )
                    except TimeoutException as e:
                        logger.error(f"Timeout exceeded while running model for {dataset} with seed {seed}, "
                                     f"params {param_set}: {e}")
                        break
                    elapsed_model_solution = time.time() - start_model_solution
                    is_converged = all(output.has_converged_map.values())

                    score_original_prefilled = partial(score_model_original,
                                                       n_nodes=n_nodes,
                                                       cg=cg,
                                                       alpha=ALPHA,
                                                       return_only_I=True)

                    sorted_arrows = sorted([(strength, arrow) for arrow, strength in output.arrow_strengths.items()],
                                           reverse=True)
                    sorted_arrows = [parse_arrow(arrow) for _, arrow in sorted_arrows]

                    start_orig_ranking = time.time()
                    original_ranking_I, best_model_original_ranking = limited_depth_search_best_model(
                        steps_ahead=steps_ahead,
                        sorted_arrows=sorted_arrows,
                        is_dag=check_arrows_dag,
                        get_score=score_original_prefilled
                    )
                    elapsed_orig_ranking = time.time() - start_orig_ranking

                    add_info = {
                        'dataset': dataset,
                        'seed': seed,
                        'n_nodes': n_nodes,
                        'n_edges': n_edges,
                        'cycle_length': param_set.cycle_length,
                        'ct_depth': param_set.ct_depth,
                        'path_length': param_set.path_length,
                        'c_set_size': param_set.c_set_size,
                        'search_depth': param_set.search_depth,
                        'elapsed_bsaf_creation': elapsed_bsaf_creation,
                        'elapsed_model_solution': elapsed_model_solution,
                        'is_converged': is_converged,
                        'fact_ranking_method': 'v2',
                        'model_ranking_method': 'original_ranking',
                        'num_edges_est': len(best_model_original_ranking),
                        'best_model': best_model_original_ranking,
                        'aba_elapsed': 0,  # No ABA elapsed time in this context
                        'ranking_elapsed': elapsed_orig_ranking,
                        'best_I': original_ranking_I,
                    }
                    logger.info(f"Run finnished: {add_info}")

                    # get and store metrics
                    best_B_est = get_matrix_from_arrow_set(best_model_original_ranking, n_nodes)
                    mt_cpdag, mt_dag = get_metrics(best_B_est, B_true)

                    # add base info
                    mt_cpdag.update(add_info)
                    mt_dag.update(add_info)

                    logger.info(f"Metrics; CPDAG: {mt_cpdag}, DAG: {mt_dag}")

                    # append to dataframes
                    cpdag_metrics_df = pd.concat([cpdag_metrics_df,
                                                  pd.DataFrame([mt_cpdag])],
                                                 ignore_index=True)
                    dag_metrics_df = pd.concat([dag_metrics_df,
                                                pd.DataFrame([mt_dag])],
                                               ignore_index=True)

                    # save to csv
                    cpdag_metrics_df.to_csv(RESULT_DIR / f'cpdag_metrics.csv', index=False)
                    dag_metrics_df.to_csv(RESULT_DIR / f'dag_metrics.csv', index=False)


if __name__ == "__main__":
    args = parse_args()

    # create results directory
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # run the experiment
    main(n_runs=args.n_runs)

    logger.info("Experiment completed successfully.")
