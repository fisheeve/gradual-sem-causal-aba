from itertools import combinations
from tqdm import tqdm
import networkx as nx
import pandas as pd

from ArgCausalDisco.utils.helpers import random_stability
from ArgCausalDisco.cd_algorithms.PC import pc
from ArgCausalDisco.utils.graph_utils import initial_strength

from src.utils.utils import get_arrows_from_model, get_matrix_from_arrow_set
from src.utils.enums import Fact, RelationEnum
from src.causal_aba.factory import ABASPSolverFactory
from src.utils.enums import SemanticEnum

from logger import logger


def get_arrow_sets_from_facts(facts, n_nodes, semantics=SemanticEnum.ST):
    # deterministically sort the facts. 
    # Even if same strength, still order is defined uniquely
    sorted_facts = sorted(facts, 
                          key=lambda x: (x.score, 
                                         x.node1, 
                                         x.node2, 
                                         str(sorted(list(x.node_set)))), 
                          reverse=True)

    # remove facts staring from weakest

    factory = ABASPSolverFactory(n_nodes=n_nodes)
    fact_idx = len(sorted_facts)

    while fact_idx >= 0:
        solver = factory.create_solver(sorted_facts[:fact_idx])
        models = solver.enumerate_extensions(semantics.value)
        only_empty_model = (models is not None
                            and len(models) == 1
                            and len(models[0].assumptions) == 0)
        break_condition = (models is not None
                           and len(models) > 0
                           and not only_empty_model
                           )
        if break_condition:
            break
        fact_idx -= 1
        logger.info(f"Trying with top {fact_idx} facts")
    arrow_sets = [get_arrows_from_model(model) for model in models]
    return arrow_sets, fact_idx


def get_arrow_sets(data,
                   seed=42,
                   alpha=0.01,
                   indep_test='fisherz',
                   uc_rule=5,
                   stable=True,
                   semantics=SemanticEnum.ST):
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

            fact = Fact(
                relation=RelationEnum(dep_type_PC),
                node1=node1,
                node2=node2,
                node_set=set(sep_set),
                score=init_strength_value
            )

            if fact not in facts:
                facts.append(fact)
    arrow_sets, num_facts = get_arrow_sets_from_facts(facts, n_nodes, semantics=semantics)

    return arrow_sets, cg, num_facts, facts


def get_best_model(models, n_nodes, cg, alpha=0.01):
    if len(models) > 50000:
        logger.info("Pick the first 50,000 models for I calculation")
        models = set(list(models)[:50000])  # Limit the number of models to 30,000

    best_model = None
    best_I = None
    best_B_est = None
    for n, model in tqdm(enumerate(models), desc="Models from ABAPC"):
        # derive B_est from the model
        B_est = get_matrix_from_arrow_set(model, n_nodes)
        G_est = nx.DiGraph(pd.DataFrame(B_est, columns=[f"X{i+1}" for i in range(B_est.shape[1])], index=[f"X{i+1}" for i in range(B_est.shape[1])]))
        est_I = 0
        for x, y in combinations(range(n_nodes), 2):
            I_from_data = list(set(cg.sepset[x, y]))
            for s, p in I_from_data:
                PC_dep_type = 'indep' if p > alpha else 'dep'
                s_text = [f"X{r+1}" for r in s]
                dep_type = 'indep' if nx.algorithms.d_separated(G_est, {f"X{x+1}"}, {f"X{y+1}"}, set(s_text)) else 'dep'
                I = initial_strength(p, len(s), alpha, 0.5, n_nodes)
                if dep_type != PC_dep_type:
                    est_I += -I
                else:
                    est_I += I
        if best_model is None or best_I < est_I:
            best_model = model
            best_I = est_I
            best_B_est = B_est
        logger.info(f"DAG from d-ABA: {best_B_est}")
    return best_model, best_B_est, best_I
