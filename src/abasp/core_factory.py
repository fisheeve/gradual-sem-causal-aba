from aspforaba.src.aspforaba import ABASolver

import src.abasp.atoms as atoms
import src.abasp.assumptions as assums
from src.abasp.utils import unique_product, powerset


class CoreABASPSolverFactory:
    """
    Factory class for creating ABASP solvers with core rules and assumptions.


    It contains all graph edge assumptions, colliders and corresponding rules.
    NOTE: It does not contain active/blocked path rules and independence assumptions.

    NOTE: Variables corresponding to graph nodes are capital letters everywhere (e.g., X, Y, Z).
    """

    def __init__(self,
                 n_nodes: int):
        self.n_nodes = n_nodes

    @staticmethod
    def _add_graph_edge_assumptions(solver, X, Y):
        for assumption in [assums.arr(X, Y), assums.arr(Y, X), assums.noe(X, Y)]:
            solver.add_assumption(assumption)
            solver.add_contrary(assumption, assums.contrary(assumption))

        for assumption1, assumption2 in unique_product([assums.arr(X, Y),
                                                        assums.arr(Y, X),
                                                        assums.noe(X, Y)], repeat=2):
            solver.add_rule(assums.contrary(assumption2), [assumption1])

        solver.add_rule(atoms.dpath(X, Y), [assums.arr(X, Y)])
        solver.add_rule(atoms.dpath(Y, X), [assums.arr(Y, X)])

        solver.add_rule(assums.edge(X, Y), [assums.arr(X, Y)])
        solver.add_rule(assums.edge(X, Y), [assums.arr(Y, X)])

    @staticmethod
    def _add_acyclicity_rules(solver, X, Y):
        solver.add_rule(assums.contrary(assums.arr(Y, X)), [atoms.dpath(X, Y)])

    @staticmethod
    def _add_non_blocking_rules(solver, X, Y, S, n_nodes):
        for N in S:
            if N not in {X, Y}:  # unique X, Y, N
                # 1) N doesn't block the S-active path between its neighbours X and Y
                #    if N is a collider and belongs to the set S
                solver.add_rule(atoms.non_blocking(N, X, Y, S), [atoms.collider(X, N, Y)])

        for N in set(range(n_nodes)) - set(S):  # nodes not in S
            if N not in {X, Y}:  # unique X, Y, N
                # 2) N doesn't block the S-active path between its neighbours X and Y
                #    if N is not a collider and doesn't belong to the set S
                solver.add_rule(atoms.non_blocking(N, X, Y, S), [atoms.not_collider(X, N, Y)])

                # 3) N doesn't block the S-active path between its neighbours X and Y
                #    if N doesn't belong to the set S and has descendant that belongs to S
                for Z in S:
                    if Z not in {X, Y, N}:
                        solver.add_rule(atoms.non_blocking(N, X, Y, S), [atoms.collider(
                            X, N, Y), atoms.descendant_of_collider(Z, X, N, Y)])

    @staticmethod
    def _add_direct_path_definition_rules(solver, X, Y, Z):
        solver.add_rule(atoms.dpath(X, Y), [assums.arr(X, Z), atoms.dpath(Z, Y)])

    @staticmethod
    def _add_collider_definition_rules(solver, X, Y, Z):
        # collider on middle node: X->Y<-Z
        solver.add_rule(atoms.collider(X, Y, Z), [assums.arr(X, Y), assums.arr(Z, Y)])

        # all not collider cases
        # X->Y->Z
        solver.add_rule(atoms.not_collider(X, Y, Z), [assums.arr(X, Y), assums.arr(Y, Z)])
        # X<-Y->Z
        solver.add_rule(atoms.not_collider(X, Y, Z), [assums.arr(Y, X), assums.arr(Y, Z)])
        # X<-Y<-Z
        solver.add_rule(atoms.not_collider(X, Y, Z), [assums.arr(Z, Y), assums.arr(Y, X)])

    @staticmethod
    def _add_collider_descendant_definition_rules(solver, X, Y, Z, N):
        solver.add_rule(atoms.descendant_of_collider(N, X, Y, Z), [atoms.collider(X, Y, Z), atoms.dpath(Y, N)])

    def create_core_solver(self):
        """
        Create a core solver for the ABASP problem.
        It contains all graph edge assumptions, colliders and corresponding rules.
        NOTE: It does not contain active/blocked path rules and independence assumptions.

        """
        solver = ABASolver()

        for X, Y in unique_product(range(self.n_nodes), repeat=2):
            self._add_acyclicity_rules(solver, X, Y)

            if X < Y:  # for X, Y unique combinations
                self._add_graph_edge_assumptions(solver, X, Y)

                for S in powerset(range(self.n_nodes)):
                    self._add_non_blocking_rules(solver, X, Y, S, self.n_nodes)

            for Z in range(self.n_nodes):
                if Z not in {X, Y}:  # X, Y, Z unique
                    self._add_direct_path_definition_rules(solver, X, Y, Z)

                    if X < Z:
                        # X < Z is to avoid duplicates as colliders are symmetric
                        self._add_collider_definition_rules(solver, X, Y, Z)

                        for N in range(self.n_nodes):
                            if N not in {X, Y, Z}:  # X, Y, Z, N unique
                                self._add_collider_descendant_definition_rules(solver, X, Y, Z, N)

        return solver
