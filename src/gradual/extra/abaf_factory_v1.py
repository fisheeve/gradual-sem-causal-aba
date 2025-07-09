import networkx as nx
from tqdm import tqdm

from typing import List
from src.utils.enums import Fact, RelationEnum
import src.causal_aba.atoms as atoms
import src.causal_aba.assumptions as assums
from src.gradual.abaf_builder import ABAFBuilder

from src.causal_aba.core_factory import CoreABASPSolverFactory
from src.gradual.abaf_builder import ABAFBuilder
from itertools import combinations
from src.utils.utils import powerset


class FactoryV1(CoreABASPSolverFactory):
    """
    Factory class for creating ABAF with active/blocked path rules and independence assumptions.
    In contrast to ABASPSolverFactory, this factory uses active path assumptions instead
    of blocked path assumptions. It also has option to add extra rules
    linking back independence to active paths and active paths to arrows.
    """

    def __init__(self, n_nodes: int):
        super().__init__(n_nodes, abaf_class=ABAFBuilder)

    @staticmethod
    def _add_path_definition_rules(solver, paths: list, X: int, Y: int):
        for path_id, my_path in enumerate(paths):
            # path definition
            solver.add_rule(atoms.path(X, Y, path_id), [assums.edge(my_path[i], my_path[i+1])
                                                        for i in range(len(my_path)-1)])

    @staticmethod
    def _add_indep_assumptions(solver, X: int, Y: int, S: set):
        solver.add_assumption(assums.indep(X, Y, S))
        solver.add_contrary(assums.indep(X, Y, S), assums.contrary(assums.indep(X, Y, S)))

    @staticmethod
    def _add_active_path_assumptions(solver, path_id: int, X: int, Y: int, S: set):
        solver.add_assumption(assums.active_path(X, Y, path_id, S))
        solver.add_contrary(assums.active_path(X, Y, path_id, S),
                            assums.contrary(assums.active_path(X, Y, path_id, S)))

    @staticmethod
    def _add_active_path_rules(solver, path_id: int, path_nodes: tuple, X: int, Y: int, S: set):
        """All nodes are non-blocking, then active path"""
        non_blocking_body = [atoms.non_blocking(path_nodes[i], path_nodes[i-1], path_nodes[i+1], S)
                             for i in range(1, len(path_nodes)-1)]
        solver.add_rule(assums.active_path(X, Y, path_id, S), [atoms.path(X, Y, path_id), *non_blocking_body])

    @staticmethod
    def _add_dependence_rules(solver, path_id: int, X: int, Y: int, S: set):
        """Active path, then dependency"""
        solver.add_rule(assums.contrary(assums.indep(X, Y, S)), [assums.active_path(X, Y, path_id, S)])

    @staticmethod
    def _add_reverse_independence_rules(solver, path_id: int, X: int, Y: int, S: set):
        """Independence, then all paths are NOT active"""
        solver.add_rule(assums.contrary(assums.active_path(X, Y, path_id, S)), [assums.indep(X, Y, S)])

    @staticmethod
    def _add_reverse_active_path_rules(solver, path_id: int, path_nodes: tuple, X: int, Y: int, S: set):
        """Active path, then v-structure at nodes included in S"""
        for i in range(1, len(path_nodes) - 1):
            if path_nodes[i] in S:
                # path_nodes[i-1] -----> path_nodes[i] <----- path_nodes[i+1]
                solver.add_rule(assums.arr(path_nodes[i-1], path_nodes[i]),
                                [assums.active_path(X, Y, path_id, S)])
                solver.add_rule(assums.arr(path_nodes[i+1], path_nodes[i]),
                                [assums.active_path(X, Y, path_id, S)])

    @staticmethod
    def _add_indep_then_noe_rule(solver, X: int, Y: int, S: set):
        """
        If X and Y are independent given S, then there is no edge between them.
        This is a rule that is not present in the original ABAF, but it can be useful
        to ensure that independence assumptions lead to no edges.
        """
        solver.add_rule(assums.noe(X, Y), [assums.indep(X, Y, S)])

    @staticmethod
    def _add_fact_strengths(solver: ABAFBuilder, facts: List[Fact]):
        """
        Add strength of facts as strengths of independence assumptions corresponding to them.
        """
        for fact in facts:
            if fact.relation == RelationEnum.indep:
                new_weight = 0.5 + 0.5*fact.score
            elif fact.relation == RelationEnum.dep:
                new_weight = 1 - (0.5 + 0.5*fact.score)
            else:
                raise ValueError(f"Unknown relation {fact.relation} for fact {fact}. "
                                 "Expected RelationEnum.indep or RelationEnum.dep.")
            # Update the weight of the independence assumption
            assumption_name = assums.indep(fact.X, fact.Y, fact.S)
            solver.update_assumption_weight(assumption_name)

    def create_solver(self, facts: List[Fact]) -> ABAFBuilder:
        '''
        Create ABAF with active paths and independence on top of the core factory.

        NOTE 1: Does not add facts by default, because this factory is used for gradual semantics where we
                assume that nothing is a 100% sure-fact.
        NOTE 2: Uses active path assumptions instead of blocked path assumptions. This is different
                from ABASPSolverFactory, which uses blocked path assumptions.
        '''
        graph = nx.complete_graph(self.n_nodes)
        solver = self.create_core_solver({})

        for X, Y in tqdm(combinations(range(self.n_nodes), 2), 
                         total=self.n_nodes*(self.n_nodes-1) // 2, 
                         desc="iterating through node combinations"):
            if X > Y:
                X, Y = Y, X

            # find all paths between X and Y
            paths = list(nx.all_simple_paths(graph, source=X, target=Y))

            # add path definition rules
            self._add_path_definition_rules(solver, paths, X, Y)

            for S in powerset(set(range(self.n_nodes)) - {X, Y}):
                # add  independence assumptions
                self._add_indep_assumptions(solver, X, Y, S)
                self._add_indep_then_noe_rule(solver, X, Y, S)
                # add active path assumptions
                for path_id, path_nodes in enumerate(paths):
                    self._add_active_path_assumptions(solver, path_id, X, Y, S)
                    # add active path rules
                    self._add_active_path_rules(solver, path_id, path_nodes, X, Y, S)
                    # add dependence rules
                    self._add_dependence_rules(solver, path_id, X, Y, S)

                    self._add_reverse_independence_rules(solver, path_id, X, Y, S)
                    # add reverse active path rules
                    self._add_reverse_active_path_rules(solver, path_id, path_nodes, X, Y, S)

        # Add facts strengths to the solver as independence assumption weights.
        if facts:
            self._add_fact_strengths(solver, facts)

        return solver
