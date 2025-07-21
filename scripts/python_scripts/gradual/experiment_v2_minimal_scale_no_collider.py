'''
Experiment with V2 scalable implementation.
Run with minimal reasonable scale for up to 20 node dataset (that is able to run on 32GB RAM).
'''

import sys
sys.path.insert(0, 'aspforaba/')
sys.path.insert(0, 'GradualABA/')
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')

import time
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from functools import partial
from typing import List, Tuple

from logger import logger

from src.abapc import get_cg_and_facts, score_model_original
from src.gradual.run import reset_weights, run_model, set_weights_according_to_facts
from src.gradual.semantic_modules.TopDiffAggregation import TopDiffAggregation
from src.gradual.search_best_model import limited_depth_search_best_model
from src.utils.bn_utils import get_dataset
from src.utils.configure_r import configure_r
from ArgCausalDisco.utils.graph_utils import is_dag
from src.utils.utils import get_matrix_from_arrow_set, parse_arrow
from ArgCausalDisco.utils.graph_utils import DAGMetrics, dag2cpdag
from ArgCausalDisco.utils.helpers import random_stability
from GradualABA.semantics.modular.LinearInfluence import LinearInfluence
from GradualABA.semantics.modular.SetMinAggregation import SetMinAggregation
from src.gradual.scalable.bsaf_builder_v2 import BSAFBuilderV2


configure_r()

ALPHA = 0.01
INDEP_TEST = 'fisherz'

SAMPLE_SIZE = 5000
RESULT_DIR = Path("./results/gradual/min_scale_v2_no_collider")

DATASETS = [
    'cancer',
    'earthquake',
    'survey',
    'asia',
    'sachs',
    'child',
    'insurance',
]

N_NODES = {'cancer':5, 
           'earthquake':5, 
           'survey':6, 
           'asia':8, 
           'sachs':11, 
           'child':20, 
           'insurance':27
}

SEED = 2024


def parse_args():
    parser = argparse.ArgumentParser(description="Run gradual ABA experiment v1")
    parser.add_argument('--n-runs', type=int, default=50, help='Number of runs for each dataset')
    parser.add_argument('--steps-ahead', type=int, default=10, help='Number of steps ahead for the search')
    return parser.parse_args()


def get_metrics(W_est, B_true):
    """
    Calculate metrics for the estimated graph W_est against the true graph B_true.
    Args:
        W_est: np.ndarray, estimated adjacency matrix
        B_true: np.ndarray, true adjacency matrix
    Returns:
        dictionary with metrics for both CPDAG and DAG representations.
    """
    B_est = (W_est != 0).astype(int)
    mt_cpdag = DAGMetrics(dag2cpdag(B_est), B_true).metrics
    B_est = (W_est > 0).astype(int)
    mt_dag = DAGMetrics(B_est, B_true).metrics
    if type(mt_cpdag['sid']) == tuple:
        mt_sid_low = mt_cpdag['sid'][0]
        mt_sid_high = mt_cpdag['sid'][1]
    else:
        mt_sid_low = mt_cpdag['sid']
        mt_sid_high = mt_cpdag['sid']
    mt_cpdag.pop('sid')
    mt_cpdag['sid_low'] = mt_sid_low
    mt_cpdag['sid_high'] = mt_sid_high

    return mt_cpdag, mt_dag

def is_dag_from_arrows(model: List[Tuple], n_nodes: int) -> bool:
    """
    Check if the model represented by tuples of nodes as arrows is a Directed Acyclic Graph (DAG).
    """
    return is_dag(get_matrix_from_arrow_set(model, n_nodes))


def main(n_runs, steps_ahead):
    facts_path = Path('./facts')
    facts_path.mkdir(parents=True, exist_ok=True)
    cpdag_metrics_df = pd.DataFrame()
    dag_metrics_df = pd.DataFrame()

    for dataset in DATASETS:
        logger.info(f"Running experiment for dataset: {dataset}")

        random_stability(SEED)
        seeds_list = np.random.randint(0, 10000, (n_runs,)).tolist()
        n_nodes = N_NODES[dataset]
        is_dag_prefilled = partial(is_dag_from_arrows, n_nodes=n_nodes)

        # create bsaf
        start_bsaf_creation = time.time()
        bsaf_builder = BSAFBuilderV2(
            n_nodes=n_nodes,
            max_cycle_size=3,
            max_collider_tree_depth = 1,  # doesn't matter when without collider tree arguments
            max_path_length = 3,
            max_conditioning_set_size = 2,
            include_collider_tree_arguments=False,
            )
        bsaf = bsaf_builder.create_bsaf()
        assumptions_dict = bsaf_builder.name_to_assumption
        elapsed_bsaf_creation = time.time() - start_bsaf_creation

        for seed in tqdm(seeds_list, desc=f"Running {dataset} with {n_runs} seeds"):
            X_s, B_true = get_dataset(dataset,
                                      seed=seed,
                                      sample_size=SAMPLE_SIZE)
            random_stability(seed)
            cg, facts = get_cg_and_facts(X_s, alpha=ALPHA, indep_test=INDEP_TEST)
            facts = sorted(facts, key=lambda x: x.score, reverse=True)  # sort by score

            # run gradual aba to get the strengths
            start_model_solution = time.time()
            reset_weights(assumptions_dict)
            set_weights_according_to_facts(assumptions_dict, facts)
            output = run_model(
                n_nodes=n_nodes,
                bsaf=bsaf,
                model_name=None,
                set_aggregation=SetMinAggregation(),
                aggregation=TopDiffAggregation(),
                influence=LinearInfluence(conservativeness=1),
                conservativeness=1,
                iterations=50,
            )
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
                is_dag=is_dag_prefilled,
                get_score=score_original_prefilled
            )
            elapsed_orig_ranking = time.time() - start_orig_ranking

            base_info = {
                'dataset': dataset,
                'seed': seed,
                'n_nodes': n_nodes,
                'elapsed_bsaf_creation': elapsed_bsaf_creation,
                'elapsed_model_solution': elapsed_model_solution,
                'is_converged': is_converged,
            }
           
            # get and store metrics
            best_B_est = get_matrix_from_arrow_set(best_model_original_ranking, n_nodes)
            mt_cpdag, mt_dag = get_metrics(best_B_est, B_true)

            add_info = {
                **base_info,
                'fact_ranking_method': 'v2',
                'model_ranking_method': 'original_ranking',
                'num_edges_est': len(best_model_original_ranking),
                'best_model': best_model_original_ranking,
                'aba_elapsed': 0, # No ABA elapsed time in this context
                'ranking_elapsed': elapsed_orig_ranking,
                'best_I': original_ranking_I,
            }

            # add base info
            mt_cpdag.update(add_info)
            mt_dag.update(add_info)

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
    main(n_runs=args.n_runs, 
         steps_ahead=args.steps_ahead)

    logger.info("Experiment completed successfully.")
