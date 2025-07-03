# Wraps the BSAF class from GradualABA to provide a simplified interface
# Takes facts and crafts assumptions and arguments
# FInally initialises GradualABA's BSAF with these assumptions and arguments
from typing import List, Dict, Tuple
import networkx as nx

import src.causal_aba.assumptions as asm
from src.utils.enums import Fact, RelationEnum
from itertools import combinations
from GradualABA.ABAF.Assumption import Assumption, Sentence
from GradualABA.BSAF.Argument import Argument
from GradualABA.BSAF.BSAF import BSAF
from GradualABA.constants import DEFAULT_WEIGHT
from logger import logger
from src.utils.utils import powerset


class NaiveCausalBSAF:
    """
    A class to create a BSAF (Bipolar Set Argumentative Framework) for causal discovery.
    The BSAF contains assumptions for arrows, no-edges, active paths, dependencies, and independencies.
    Arguments are the following:
    - Active path implies v-structure. More specifically the two arrows leading to the middle node.
    - Dependency implies active path (weak evidence). Weak evidence is ensured by scaling the strength of the dependency.
    - Independence implies blocked path.
    - Independence implies no-edge.

    All assumptions have initial strength of 0.5, 
    except for dep and indep assumptions:
        - Independency assumptions are scaled to [0.5, 1.0] based on the score. Strength = 0.5 + 0.5 * fact_score
        - Dependency assumptions are scaled to [0.5, 1.0] based on the score, also take into account that there 
            are multiple possible active paths as follows.  Strength = 0.5 + 0.5 * (fact_score / num_relevant_paths)
    """
    def __init__(self, n_nodes: int, facts: List[Fact], default_weight: float = DEFAULT_WEIGHT):
        self.n_nodes = n_nodes
        self.facts = facts
        self.default_weight = default_weight
        self.bsaf = None
        self.arguments = set()
        self.assumptions = set()
        self.name2sent = dict()

    def _add_assumption(self, name: str, initial_weight: float):
        if name not in self.assumptions:
            self.name2sent[name] = Assumption(name=name,
                                              contrary=asm.contrary(name),
                                              initial_weight=initial_weight)
            self.name2sent[asm.contrary(name)] = Sentence(name=asm.contrary(name))

            self.assumptions.add(self.name2sent[name])
        else:
            logger.warning(f"Assumption {name} already exists, skipping creation.")

    def _add_arrow_assumptions(self):
        """Add arrow assumptions corresponding to the edges of the graph
        """
        for node1, node2 in combinations(range(self.n_nodes), 2):
            self._add_assumption(asm.arr(node1, node2),
                                 initial_weight=self.default_weight)
            self._add_assumption(asm.arr(node2, node1),
                                 initial_weight=self.default_weight)
            self._add_assumption(asm.noe(node1, node2),
                                 initial_weight=self.default_weight)

    def _add_active_path_assumptions(self, paths: Dict[Tuple, List[Tuple]]):
        """
        Add active path assumptions corresponding to the paths in the graph

        :param paths: Dictionary where keys are tuples of node pairs (X, Y) and values are lists of paths.
            Paths mentioned here are tuples of integers representing node indices on the path.
        """
        for node1, node2 in combinations(range(self.n_nodes), 2):
            if node1 > node2:
                node1, node2 = node2, node1

            for S in powerset(set(range(self.n_nodes)) - {node1, node2}):
                for path_id, path_nodes in enumerate(paths.get((node1, node2), [])):
                    self._add_assumption(asm.active_path(node1, node2, path_id, S),
                                         initial_weight=self.default_weight)
                    for i in range(1, len(path_nodes) - 1):
                        if path_nodes[i] in S:
                            # v-structure argument
                            arg1 = Argument(
                                name='active_path_implies_v_structure',
                                claim=self.name2sent[asm.arr(path_nodes[i-1], path_nodes[i])],
                                premise=[self.name2sent[asm.active_path(node1, node2, path_id, S)]]
                            )
                            arg2 = Argument(
                                name='active_path_implies_v_structure',
                                claim=self.name2sent[asm.arr(path_nodes[i+1], path_nodes[i])],
                                premise=[self.name2sent[asm.active_path(node1, node2, path_id, S)]]
                            )
                            self.arguments.update({arg1, arg2})

    def _add_dep_and_indep_assumptions(self, paths: Dict[Tuple, List[Tuple]]):
        """Add dependency and independence assumptions based on the facts provided
        """
        for fact in self.facts:

            if fact.relation == RelationEnum.dep:
                num_relevant_paths = len(paths.get((fact.node1, fact.node2), []))
                # distribute the support between all relevant paths
                distributed_score = fact.score / num_relevant_paths if num_relevant_paths > 0 else scaled_score
                # scale to ensure score is within [0.5, 1.0]
                scaled_score = 0.5 + 0.5 * distributed_score
                self._add_assumption(asm.dep(fact.node1, fact.node2, fact.node_set),
                                     initial_weight=scaled_score)
                for path_id in range(num_relevant_paths):
                    # weak evidence argument for the active path
                    argument = Argument(
                        name='dep_weekly_implies_active_path',
                        claim=self.name2sent[asm.active_path(fact.node1, fact.node2, path_id, fact.node_set)],
                        premise=[self.name2sent[asm.dep(fact.node1, fact.node2, fact.node_set)]]
                    )
                    self.arguments.add(argument)
            elif fact.relation == RelationEnum.indep:
                scaled_score = 0.5 + 0.5 * fact.score
                self._add_assumption(asm.indep(fact.node1, fact.node2, fact.node_set),
                                     initial_weight=scaled_score)
                for path_id in range(len(paths.get((fact.node1, fact.node2), []))):
                    # strong evidence argument against the active path
                    argument1 = Argument(
                        name='indep_implies_blocked_path',
                        claim=self.name2sent[asm.contrary(asm.active_path(
                            fact.node1, fact.node2, path_id, fact.node_set))],
                        premise=[self.name2sent[asm.indep(fact.node1, fact.node2, fact.node_set)]]
                    )
                    # evidence for no-edge assumption
                    argument2 = Argument(
                        name='indep_implies_no_edge',
                        claim=self.name2sent[asm.noe(fact.node1, fact.node2)],
                        premise=[self.name2sent[asm.indep(fact.node1, fact.node2, fact.node_set)]]
                    )
                    self.arguments.update({argument1, argument2})

    def _build_arguments(self):
        self._add_arrow_assumptions()

        graph = nx.complete_graph(self.n_nodes)
        all_paths = dict()
        for node1, node2 in combinations(range(self.n_nodes), 2):
            if node1 > node2:
                node1, node2 = node2, node1
            all_paths[(node1, node2)] = [tuple(p) for p in nx.all_simple_paths(graph, source=node1, target=node2)]

        self._add_active_path_assumptions(all_paths)
        self._add_dep_and_indep_assumptions(all_paths)

    def create_bsaf(self):
        """
        Create the BSAF with the assumptions and arguments built from the facts.
        """
        self._build_arguments()

        # Create BSAF with the assumptions and arguments
        self.bsaf = BSAF(assumptions=self.assumptions,
                         arguments=self.arguments)

        return self.bsaf
