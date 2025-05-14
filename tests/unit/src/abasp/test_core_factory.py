from src.abasp.core_factory import CoreABASPSolverFactory
from src.abasp.utils import get_arrows_from_model


def test_all_3_node_graphs():
    expected = set([
        frozenset(),
        frozenset({(1, 0)}),
        frozenset({(0, 2)}),
        frozenset({(0, 1)}),
        frozenset({(1, 2)}),
        frozenset({(2, 0)}),
        frozenset({(2, 1)}),
        # chains
        frozenset({(0, 2), (2, 1)}),
        frozenset({(0, 1), (2, 0)}),
        frozenset({(1, 0), (2, 1)}),
        frozenset({(1, 2), (2, 0)}),
        frozenset({(1, 0), (0, 2)}),
        frozenset({(0, 1), (1, 2)}),
        # colliders
        frozenset({(0, 1), (2, 1)}),
        frozenset({(0, 2), (1, 2)}),
        frozenset({(1, 0), (2, 0)}),
        # confounders
        frozenset({(2, 0), (2, 1)}),
        frozenset({(1, 0), (1, 2)}),
        frozenset({(0, 1), (0, 2)}),
        # three arrows configurations
        frozenset({(0, 1), (0, 2), (1, 2)}),
        frozenset({(0, 1), (0, 2), (2, 1)}),
        frozenset({(1, 0), (0, 2), (1, 2)}),
        frozenset({(1, 0), (2, 0), (2, 1)}),
        frozenset({(0, 1), (2, 0), (2, 1)}),
        frozenset({(1, 0), (1, 2), (2, 0)})
    ])
    factory = CoreABASPSolverFactory(3)
    solver = factory.create_core_solver(facts=[])
    models = solver.get_stable_models()
    arrow_sets = [frozenset(get_arrows_from_model(model)) for model in models]

    assert len(arrow_sets) == len(expected)
    assert set(arrow_sets) == expected
