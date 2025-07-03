import sys  # noqa
sys.path.append("GradualABA")  # noqa

from src.gradual.naive.causal_bsaf import NaiveCausalBSAF
from src.utils.enums import Fact, RelationEnum
import pytest


@pytest.fixture
def bsaf_fixture():
    facts = [
        Fact(relation=RelationEnum.indep,
             node1=0,
             node2=1,
             node_set={},
             score=1.0),
        Fact(relation=RelationEnum.dep,
             node1=1,
             node2=2,
             node_set={0},
             score=1.0),

    ]
    return NaiveCausalBSAF(n_nodes=3, facts=facts)


def test_creates_correct_number_of_arrow_assumptions(bsaf_fixture: NaiveCausalBSAF):
    bsaf_fixture._add_arrow_assumptions()
    # 3 nodes, arr_xy, arr_yx, noe_xy for each pair (x, y), so 9 in total
    assert len(bsaf_fixture.assumptions) == 9


def test_creates_correct_number_of_arguments(bsaf_fixture: NaiveCausalBSAF):
    bsaf_fixture._build_arguments()
    # sort arguments by their names
    args_dict = dict()
    for arg in bsaf_fixture.arguments:
        args_dict[arg.name] = args_dict.get(arg.name, [])
        args_dict[arg.name].append(arg)

    dep_to_ap = "dep_weekly_implies_active_path"
    indep_to_bp = "indep_implies_blocked_path"
    indep_to_noe = "indep_implies_no_edge"
    ap_to_v_struct = "active_path_implies_v_structure"

    assert set(args_dict.keys()) == {dep_to_ap, indep_to_bp, indep_to_noe, ap_to_v_struct}
    assert len(args_dict[dep_to_ap]) == 2  # one dep fact, 2 paths for the node pair
    assert len(args_dict[indep_to_bp]) == 2  # one indep fact, 2 paths for the node pair
    assert len(args_dict[indep_to_noe]) == 1  # one indep fact
    assert len(args_dict[ap_to_v_struct]) == 6  # 3 possible v_structures, 2 arguments for each v-structure


def test_support_and_attack_sets_are_correct(bsaf_fixture: NaiveCausalBSAF):
    bsaf = bsaf_fixture.create_bsaf()

    # Check indep and dep assumptions have no support or attack sets
    for asm in bsaf.assumptions:
        if asm.name.startswith("indep") or asm.name.startswith("dep"):
            assert len(bsaf.supports[asm]) == 0
            assert len(bsaf.attacks[asm]) == 0

    # Check active path assumptions are supported by indep and dep assumptions
    all_ap_supports, all_ap_attacks = set(), set()
    for asm in bsaf.assumptions:
        if asm.name.startswith("active_path"):
            all_ap_supports.update(bsaf.supports[asm])
            all_ap_attacks.update(bsaf.attacks[asm])
    indep_assumption = next(asm for asm in bsaf.assumptions if asm.name.startswith("indep"))
    dep_assumption = next(asm for asm in bsaf.assumptions if asm.name.startswith("dep"))
    assert all_ap_supports == {frozenset({dep_assumption})}
    assert all_ap_attacks == {frozenset({indep_assumption})}

    # check arrows are supported by active paths
    all_arrow_supports, all_arrow_attacks = set(), set()
    for asm in bsaf.assumptions:
        if asm.name.startswith("arr"):
            all_arrow_supports.update(bsaf.supports[asm])
            all_arrow_attacks.update(bsaf.attacks[asm])
    assert len(all_arrow_attacks) == 0
    assert all(asm.name.startswith("active_path") for asm_set in all_arrow_supports for asm in asm_set)

    # check no_edge assumptions are supported by independence assumptions
    all_noe_supports, all_noe_attacks = set(), set()
    for asm in bsaf.assumptions:
        if asm.name.startswith("noe"):
            all_noe_supports.update(bsaf.supports[asm])
            all_noe_attacks.update(bsaf.attacks[asm])
    assert all_noe_supports == {frozenset({indep_assumption})}
    assert len(all_noe_attacks) == 0
