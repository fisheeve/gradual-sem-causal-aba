from itertools import chain, combinations, product
from enum import Enum
from dataclasses import dataclass


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

    @classmethod
    def from_tuple(cls, tpl):
        node1, node_set_tuple, node2, relation, _, score = tpl
        # independence is symmetric
        if node1 > node2:
            node1, node2 = node2, node1
        return cls(
            relation=RelationEnum(relation),
            node1=int(node1),
            node2=int(node2),
            node_set=set(int(i) for i in node_set_tuple),
            score=float(score)
        )
