'''
1. Experiment providing these indep fact strengths to rank models
    - Use original facts valuation
    - Use refined fact strengths only consider independence facts tho.
    - Or use arrow and noe strengths.
2. Experiment providing these indep fact strengths to rank initial facts as well
    - Only consider independence facts tho

6 methods to accomplish the above
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
from dataclasses import asdict
from tqdm import tqdm

from logger import logger

import src.causal_aba.assumptions as assums
from src.abapc import get_best_model_various_valuations, get_cg_and_facts
from src.gradual.abaf_opt import ABAFOptimised
from src.gradual.extra.abaf_factory_v1 import FactoryV1
from src.gradual.run import reset_weights, run_get_bsaf_and_assum_dict, run_model, set_weights_according_to_facts
from src.gradual.semantic_modules.TopDiffAggregation import TopDiffAggregation
from src.utils.bn_utils import get_dataset
from src.utils.enums import Fact, RelationEnum
from src.utils.fact_utils import get_fact_location
from src.utils.configure_r import configure_r
from ArgCausalDisco.causalaba import CausalABA
from ArgCausalDisco.utils.graph_utils import DAGMetrics, dag2cpdag, set_of_models_to_set_of_graphs
from ArgCausalDisco.utils.helpers import random_stability
from GradualABA.semantics.modular.LinearInfluence import LinearInfluence
from GradualABA.semantics.modular.SetMinAggregation import SetMinAggregation


configure_r()

ALPHA = 0.01
INDEP_TEST = 'fisherz'

SAMPLE_SIZE = 5000
RESULT_DIR = Path("./results/gradual/experiment_v1")

DATASETS = ['cancer',
            'earthquake',
            'survey']
N_NODES = {
    'cancer': 5,
    'earthquake': 5,
    'survey': 6,
}
SEED = 2024


def parse_args():
    parser = argparse.ArgumentParser(description="Run gradual ABA experiment v1")
    parser.add_argument('--n-runs', type=int, default=50, help='Number of runs for each dataset')
    return parser.parse_args()


def get_models_from_facts(facts, seed, n_nodes, base_location='./facts.lp'):
    """ Get models from facts using the CausalABA framework.
    Args:
        facts: list of Fact objects
        seed: int, random seed for reproducibility
        n_nodes: int, number of nodes in the causal graph
        base_location: str, base path for saving facts
    Returns:
        model_sets: list of models, where each model is a frozenset of node pair tuples 
        representing arrows in the causal graph."""
    facts_location = get_fact_location(facts, base_location=base_location)

    random_stability(seed)
    model_sets, _ = CausalABA(
        n_nodes, facts_location, weak_constraints=True, skeleton_rules_reduction=True,
        fact_pct=1.0, search_for_models='first',
        opt_mode='optN', print_models=False, set_indep_facts=False)
    model_sets, _ = set_of_models_to_set_of_graphs(model_sets, n_nodes, False)
    return model_sets


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


def main(n_runs):
    facts_path = Path('./facts')
    facts_path.mkdir(parents=True, exist_ok=True)
    cpdag_metrics_df = pd.DataFrame()
    dag_metrics_df = pd.DataFrame()

    for dataset in DATASETS:
        logger.info(f"Running experiment for dataset: {dataset}")

        random_stability(SEED)
        seeds_list = np.random.randint(0, 10000, (n_runs,)).tolist()
        n_nodes = N_NODES[dataset]

        # create bsaf
        start_bsaf_creation = time.time()
        factory = FactoryV1(n_nodes=n_nodes)
        bsaf, assumptions_dict = run_get_bsaf_and_assum_dict(
            factory=factory,
            facts=[],
            set_aggregation=SetMinAggregation(),
            abaf_class=ABAFOptimised,)
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

            # original
            base_location = str(facts_path / f'{dataset}_{seed}_facts.lp')

            start_causal_aba = time.time()
            original_model_sets = get_models_from_facts(facts, seed, n_nodes, base_location=base_location)
            elapsed_causal_aba = time.time() - start_causal_aba

            original_best_model_collection = get_best_model_various_valuations(
                models=original_model_sets,
                n_nodes=n_nodes,
                cg=cg,
                alpha=ALPHA,
                indep_to_strength=output.indep_strengths,
                arr_strength=output.arrow_strengths,
            )

            # with refined independence fact strengths
            only_indep_facts = [fact for fact in facts if fact.relation == RelationEnum.indep]
            # Modify strengths
            for fact in only_indep_facts:
                fact.score = output.indep_strengths[assums.indep(fact.node1, fact.node2, fact.node_set)]

            base_location = str(facts_path / f'{dataset}_{seed}_refined_facts_indep.lp')

            start_refined_causal_aba = time.time()
            refined_indep_model_sets = get_models_from_facts(
                only_indep_facts, seed, n_nodes, base_location=base_location)
            elapsed_refined_causal_aba = time.time() - start_refined_causal_aba

            refined_indep_best_model_collection = get_best_model_various_valuations(
                models=refined_indep_model_sets,
                n_nodes=n_nodes,
                cg=cg,
                alpha=ALPHA,
                indep_to_strength=output.indep_strengths,
                arr_strength=output.arrow_strengths,
            )

            base_info = {
                'dataset': dataset,
                'seed': seed,
                'n_nodes': n_nodes,
                'elapsed_bsaf_creation': elapsed_bsaf_creation,
                'elapsed_model_solution': elapsed_model_solution,
                'is_converged': is_converged,
            }

            # Now let's evaluate the model collections
            for model_collection, fact_ranking_method, aba_elaplsed in [
                (original_best_model_collection, 'original', elapsed_causal_aba),
                (refined_indep_best_model_collection, 'refined_indep_facts', elapsed_refined_causal_aba)
            ]:
                for model_ranking_method, best_model_object in asdict(model_collection).items():
                    best_model = best_model_object['best_model']
                    best_B_est = best_model_object['best_B_est']
                    best_I = best_model_object['best_I']
                    ranking_elapsed = best_model_object['elapsed']

                    # get metrics
                    mt_cpdag, mt_dag = get_metrics(best_B_est, B_true)

                    add_info = {
                        **base_info,
                        'fact_ranking_method': fact_ranking_method,
                        'model_ranking_method': model_ranking_method,
                        'aba_elapsed': aba_elaplsed,
                        'ranking_elapsed': ranking_elapsed,
                        'num_edges_est': len(best_model),
                        'best_model': best_model,
                        'best_I': best_I,
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
    n_runs = args.n_runs

    # create results directory
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # run the experiment
    main(n_runs)

    logger.info("Experiment completed successfully.")
