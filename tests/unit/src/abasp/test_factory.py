import pytest
from tests.utils import facts_from_file
from src.abasp.factory import ABASPSolverFactory
from src.abasp.utils import get_arrows_from_model


@pytest.mark.parametrize(
    "filepath, n_nodes, assert_equal, expected",
    [('tests/data/five_node_colombo_example.lp', 5, True, {(0, 2), (1, 2), (0, 4), (2, 4), (3, 4), (0, 3), (1, 4), (2, 3)}),
     ('tests/data/five_node_sprinkler_example.lp', 5, False, {(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)}),
     ('tests/data/six_node_example.lp', 6, True, {(2, 4), (1, 2), (0, 4), (1, 5), (0, 3), (2, 3), (0, 2), (4, 5), (0, 5), (2, 5), (3, 5)})],
    ids=['colombo', 'sprinkler', 'six_node'])
def test_graph_examples(filepath, n_nodes, assert_equal, expected):
    facts = facts_from_file(filepath)
    factory = ABASPSolverFactory(n_nodes)
    solver = factory.create_solver(facts=facts)
    models = solver.get_stable_models()
    arrow_sets = [get_arrows_from_model(model) for model in models]

    assert expected in arrow_sets

    if assert_equal:
        assert len(arrow_sets) == 1
