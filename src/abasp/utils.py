import numpy as np
from itertools import chain, combinations, product
from enum import Enum
from dataclasses import dataclass
from aspforaba.src.aspforaba.abaf import AssumptionSet


def is_unique(ary):
    return len(ary) == len(set(ary))


def powerset(s):
    s = sorted(list(s))
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def unique_product(elements, repeat: int):
    '''
    Generate all combinations of elements without duplicates.
    '''
    for element_set in product(elements, repeat=repeat):
        if is_unique(element_set):
            yield element_set


def parse_arrow(arrow: str):
    """
    Parse an arrow string into a tuple of integers.
    :param arrow: The arrow string to parse.
    :return: A tuple of integers representing the arrow.
    """

    try:
        _, node1, node2 = arrow.split("_")
        node1 = int(node1)
        node2 = int(node2)
    except ValueError as e:
        raise ValueError(f"Arrow must be of form arr_<node1>_<node2>, got {arrow}. Here is the exception {e}")

    return node1, node2


def get_arrows_from_model(model: AssumptionSet):
    """
    Get the arrows from a model.
    :param model: The model to extract arrows from.
    :return: A set of tuples representing the arrows in the model.
    """
    arrows = set()
    for assumption in model.assumptions:
        if assumption.startswith("arr_"):
            node1, node2 = parse_arrow(assumption)
            arrows.add((node1, node2))
    return arrows


def get_graph_matrix(n_nodes, arrows):
    """
    Get the adjacency matrix of a graph.
    :param n_nodes: The number of nodes in the graph.
    :param arrows: The arrows in the graph.
    :return: A numpy array representing the adjacency matrix of the graph.
    """
    matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    for node1, node2 in arrows:
        matrix[node1, node2] = 1
    return matrix


class RelationEnum(str, Enum):
    dep = "dep"
    indep = "indep"


@dataclass
class Fact:
    relation: RelationEnum
    node1: int
    node2: int
    node_set: set
    score: float
