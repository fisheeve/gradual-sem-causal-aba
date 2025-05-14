from pathlib import Path
from itertools import combinations

from ArgCausalDisco.utils.helpers import random_stability
from ArgCausalDisco.utils.data_utils import load_bnlearn_data_dag
from ArgCausalDisco.cd_algorithms.PC import pc
from ArgCausalDisco.utils.graph_utils import initial_strength

from src.abasp.utils import Fact, RelationEnum, get_arrows_from_model
from src.abasp.factory import ABASPSolverFactory


def get_dataset(dataset_name='cancer',
                seed=42,
                sample_size=5000,
                data_path=Path(__file__).resolve().parents[1] / 'ArgCausalDisco' / 'datasets'):
    X_s, B_true = load_bnlearn_data_dag(dataset_name,
                                        data_path,
                                        sample_size,
                                        seed=seed,
                                        print_info=True,
                                        standardise=True)
    return X_s, B_true


def get_stable_arrow_sets(data,
                          seed=42,
                          alpha=0.01,
                          indep_test='fisherz',
                          uc_rule=5,
                          stable=True):
    """
    Get the stable models from the ABAPC algorithm
    Args:
        X_s: np.array
            The dataset to be used for the ABAPC algorithm
        seed: int
            The seed to be used for the random number generator
    Returns:
        models: list
            The stable models from the ABAPC algorithm
            in a form of arrow sets.
    """
    random_stability(seed)
    n_nodes = data.shape[1]
    cg = pc(data=data, alpha=alpha, indep_test=indep_test, uc_rule=uc_rule,
            stable=stable, show_progress=True, verbose=True)
    facts = []

    for node1, node2 in combinations(range(n_nodes), 2):
        test_PC = [t for t in cg.sepset[node1, node2]]
        for sep_set, p in test_PC:
            dep_type_PC = "indep" if p > alpha else "dep"
            init_strength_value = initial_strength(p, len(sep_set), alpha, 0.5, n_nodes)
            if node1 > node2:
                node1, node2 = node2, node1

            fact = Fact(
                relation=RelationEnum(dep_type_PC),
                node1=node1,
                node2=node2,
                node_set=set(sep_set),
                score=init_strength_value
            )

            if fact not in facts:
                facts.append(fact)

    sorted_facts = sorted(facts, key=lambda x: x.score, reverse=True)

    # binary search to find the largest fact set where stable extensions exist

    left_idx = 0
    right_idx = len(sorted_facts) - 1
    factory = ABASPSolverFactory(n_nodes=n_nodes)

    result_table = dict()  # fact_idx: result, result is True if extension was found

    def get_extensions(factory: ABASPSolverFactory, facts, result_table):
        index = len(facts) - 1

        if index not in result_table:
            solver = factory.create_solver(facts)
            models = solver.get_stable_models()
            result_table[index] = models if models is not None else []

        return result_table[index]

    final_extensions = None
    final_facts = None
    final_fact_index = None

    while left_idx <= right_idx:
        mid = (left_idx + right_idx) // 2

        exts_mid = get_extensions(factory, sorted_facts[:mid + 1], result_table)
        exts_mid_next = get_extensions(factory, sorted_facts[:mid + 2], result_table)

        if len(exts_mid) == 0:  # overshoot
            right_idx = mid - 1
        elif len(exts_mid_next) > 0:  # undershoot
            left_idx = mid + 1
        else:
            # mid is the largest index where extensions exist
            final_extensions = exts_mid
            # TODO: log final facts, index and extensions
            final_facts = sorted_facts[:mid + 1]
            final_fact_index = mid
            break

    arrow_sets = [get_arrows_from_model(model) for model in final_extensions]
    return arrow_sets
