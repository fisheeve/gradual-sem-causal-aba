from typing import List
import networkx as nx

import src.abasp.atoms as atoms
import src.abasp.assumptions as assums

from src.abasp.core_factory import CoreABASPSolverFactory
from src.abasp.utils import powerset, Fact, RelationEnum
from itertools import combinations


class ABASPSolverFactory(CoreABASPSolverFactory):
    """
    Factory class for creating ABASP solvers with active/blocked path rules and independence assumptions.
    """

    def __init__(self, n_nodes: int):
        super().__init__(n_nodes)

    @staticmethod
    def _add_path_definition_rules(solver, paths, X, Y):
        for path_id, my_path in enumerate(paths):
            # path definition
            solver.add_rule(atoms.path(X, Y, path_id), [assums.edge(my_path[i], my_path[i+1])
                                                        for i in range(len(my_path)-1)])

    @staticmethod
    def _add_indep_assumptions(solver, X, Y, S):
        solver.add_assumption(assums.indep(X, Y, S))
        solver.add_contrary(assums.indep(X, Y, S), assums.contrary(assums.indep(X, Y, S)))

    @staticmethod
    def _add_independence_rules(solver, paths, X, Y, S):
        indep_body = [assums.blocked_path(X, Y, path_id, S) for path_id in range(len(paths))]
        if len(indep_body) > 0:  # avoid adding duplicate rule to the facts
            solver.add_rule(assums.indep(X, Y, S), indep_body)

    @staticmethod
    def _add_blocked_path_assumptions(solver, path_id, X, Y, S):
        # active path definition
        solver.add_assumption(assums.blocked_path(X, Y, path_id, S))
        solver.add_contrary(assums.blocked_path(X, Y, path_id, S),
                            assums.contrary(assums.blocked_path(X, Y, path_id, S)))

    @staticmethod
    def _add_active_path_rules(solver, path_id, path_nodes, X, Y, S):
        non_blocking_body = [atoms.non_blocking(path_nodes[i], path_nodes[i-1], path_nodes[i+1], S)
                             for i in range(1, len(path_nodes)-1)]
        solver.add_rule(assums.active_path(X, Y, path_id, S), [atoms.path(X, Y, path_id), *non_blocking_body])

    @staticmethod
    def _add_dependence_rules(solver, path_id, X, Y, S):
        solver.add_rule(assums.dep(X, Y, S), [assums.active_path(X, Y, path_id, S)])

    def create_solver(self, facts: List[Fact]):
        '''
        Create ABASP solver with active/blocked path rules, 
        independence assumptions and facts.

        NOTE 1: add ap assumption and rule only for dep and indep present in facts
        NOTE 2: don't consider paths that contain edges where nodes are independent according to external facts

        '''
        solver = self.create_core_solver()

        # not consider paths that have edges with nodes that are independent (for any set S)
        # edges_to_remove = set()
        # for fact in facts:
        #     if fact.relation == RelationEnum.indep:
        #         edges_to_remove.add((fact.node1, fact.node2))

        graph = nx.complete_graph(self.n_nodes)
        # remove edges that are independent according to external facts
        # graph.remove_edges_from(edges_to_remove)

        for fact in facts:
            X, Y, S = fact.node1, fact.node2, fact.node_set
            paths = [tuple(p) for p in nx.all_simple_paths(graph, source=X, target=Y)]
            self._add_path_definition_rules(solver, paths, X, Y)

            # add assumption
            self._add_indep_assumptions(solver, X, Y, S)

            if fact.relation == RelationEnum.dep:
                # add dependency fact
                solver.add_rule(assums.dep(fact.node1, fact.node2, fact.node_set), [])
                # add corresponding independence rules to attack
                for path_id, my_path in enumerate(paths):
                    # active path definition
                    self._add_blocked_path_assumptions(solver, path_id, X, Y, S)
                    # active path rule
                    self._add_active_path_rules(solver, path_id, my_path, X, Y, S)
                self._add_independence_rules(solver, paths, X, Y, S)
            elif fact.relation == RelationEnum.indep:
                # add independence fact
                solver.add_rule(assums.indep(fact.node1, fact.node2, fact.node_set), [])
                # add corresponding dependency rules to attack
                for path_id, my_path in enumerate(paths):
                    self._add_active_path_rules(solver, path_id, my_path, X, Y, S)
                    self._add_dependence_rules(solver, path_id, X, Y, S)

        return solver
