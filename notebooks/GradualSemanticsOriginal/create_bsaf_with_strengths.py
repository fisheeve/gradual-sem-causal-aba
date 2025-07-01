import sys
import pickle
import time
sys.path.insert(0, '../../')

sys.path.insert(0, '../../ArgCausalDisco/')
sys.path.insert(0, '../../notears/')
sys.path.append("../../GradualABA")

import time
from pathlib import Path
from src.utils.enums import Fact, RelationEnum
from src.utils.bn_utils import get_dataset
from ArgCausalDisco.utils.helpers import random_stability
from itertools import combinations

from ArgCausalDisco.cd_algorithms.PC import pc
from ArgCausalDisco.utils.graph_utils import initial_strength
from src.causal_aba.assumptions import indep

from src.causal_aba.factory import ABASPSolverFactory

from GradualABA.ABAF import ABAF
from GradualABA.semantics.bsafDiscreteModular import DiscreteModular
from GradualABA.semantics.modular.LinearInfluence import LinearInfluence
from GradualABA.semantics.modular.ProductAggregation import ProductAggregation
from GradualABA.semantics.modular.SetProductAggregation import SetProductAggregation
from GradualABA.semantics.modular import SumAggregation
from GradualABA.semantics.modular import QuadraticMaximumInfluence


ALPHA = 0.01
INDEP_TEST = 'fisherz'
# DATASETS = [
#     'cancer',
    # 'earthquake',
    # 'survey',
    # 'asia'
# ]

N_RUNS = 50
SAMPLE_SIZE = 5000
RESULT_DIR = Path("./results/")

RESULT_DIR.mkdir(parents=True, exist_ok=True)

DELTA = 5
EPSILON = 1e-3
LOAD_FROM_FILE = False

dataset_name = 'cancer'
seed = 2024
appendix = f"{dataset_name}_with_strengths"

### ----------------------------- Convert ABAF to BSAF -----------------------------

if LOAD_FROM_FILE and Path(RESULT_DIR / f"bsaf_{appendix}.pkl").exists():
    with open(RESULT_DIR / f"bsaf_{appendix}.pkl", "rb") as f:
        bsaf = pickle.load(f)
    print("Loaded BSAF from file.")
else:
    X_s, B_true = get_dataset(dataset_name,
                        seed=seed,
                        sample_size=SAMPLE_SIZE)

    # get facts from pc
    uc_rule = 5
    data = X_s

    random_stability(seed)
    n_nodes = data.shape[1]
    cg = pc(data=data, alpha=ALPHA, indep_test=INDEP_TEST, uc_rule=uc_rule,
            stable=True, show_progress=True, verbose=True)
    facts = []

    for node1, node2 in combinations(range(n_nodes), 2):
        test_PC = [t for t in cg.sepset[node1, node2]]
        for sep_set, p in test_PC:
            dep_type_PC = "indep" if p > ALPHA else "dep"
            init_strength_value = initial_strength(p, len(sep_set), ALPHA, 0.5, n_nodes)

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
    pickle.dump(sorted_facts, open(RESULT_DIR / f"facts_{appendix}.pkl", "wb"))

    print(f"Facts from PC: {len(sorted_facts)}")

    factory = ABASPSolverFactory(n_nodes=n_nodes, optimise_remove_edges=False)
    solver = factory.create_solver(sorted_facts)

    pickle.dump(solver, open(RESULT_DIR / f"solver_{appendix}.pkl", "wb"))

    iccma_input = f"p aba {len(solver.abaf.atom_to_idx)}\n"

    for assumption in solver.assumptions:
        a_id = solver.abaf.atom_to_idx[assumption]
        iccma_input += f"a {a_id}\n"
    for assumption, contrary_set in solver.contraries.items():
        contrary = contrary_set.pop()  # assuming there's only one contrary per assumption
        c_id = solver.abaf.atom_to_idx[contrary]
        a_id = solver.abaf.atom_to_idx[assumption]
        iccma_input += f"c {a_id} {c_id}\n"
    for head, body in solver.rules:
        head_id = solver.abaf.atom_to_idx[head]
        body_ids = [solver.abaf.atom_to_idx[atom] for atom in body]
        iccma_input += f"r {head_id} {' '.join(map(str, body_ids))}\n"

    # print(iccma_input)

    iccma_input_path = Path(RESULT_DIR / f"iccma_input_{appendix}.aba")
    with open(iccma_input_path, 'w') as f:
        f.write(iccma_input)



    ### ----------------------------- Load ABAF from ICCMA example -----------------------------
    abaf = ABAF(path=iccma_input_path)
    
    # Manualy give strengths to independence assumptions
    idx_to_assumption = {
        asmp.name: asmp for asmp in abaf.assumptions
    }

    for fact in sorted_facts:
        asmp_name = indep(fact.node1, fact.node2, fact.node_set)
        asmp_index = solver.abaf.asmpt_to_idx[asmp_name]

        # can't attack with indep-contarary atoms, so have do ignore dependence facts :(
        if fact.relation == RelationEnum.indep:
            idx_to_assumption[str(asmp_index)].initial_weight = fact.score


    bsaf = abaf.to_bsaf()

    pickle.dump(bsaf, open(RESULT_DIR / f"bsaf_{appendix}.pkl", "wb"))
