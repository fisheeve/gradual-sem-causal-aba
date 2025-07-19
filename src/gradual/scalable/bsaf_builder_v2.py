from typing import List, Dict, Tuple, Set
from src.utils.enums import Fact
from src.constants import DEFAULT_WEIGHT  # 0.5
import src.causal_aba.assumptions as asm
from GradualABA.ABAF.Assumption import Assumption, Sentence
from GradualABA.BSAF.Argument import Argument
from GradualABA.ABAF.Rule import Rule
from GradualABA.BSAF.BSAF import BSAF
from itertools import combinations, permutations

from logger import logger


class BSAFBuilderV2:
    def __init__(self,
                 n_nodes: int,
                 default_weight: float = DEFAULT_WEIGHT,
                 max_cycle_size: int = 5,
                 max_collider_tree_depth: int = 5,
                 max_path_length: int = 5,
                 max_conditioning_set_size: int = 5):
        self.n_nodes = n_nodes
        self.default_weight = default_weight
        self.arguments = set()
        Rule.reset_identifiers()
        Sentence.reset_identifiers()
        Assumption.reset_identifiers()
        self.name_to_assumption: Dict[str, Assumption] = dict()  # Maps assumption names to Assumption objects
        # Maps names to Sentence objects
        # Sentences can be assumptions or assumption contrary literals
        self.name_to_sentence: Dict[str, Sentence] = dict()

        # Longest cycle can be of size n_nodes, so we set reasonable limit
        self.max_cycle_size = min(max_cycle_size, n_nodes)
        # deepest collider tree can be of size n_nodes-1, so we set reasonable limit
        self.max_collider_tree_depth = min(max_collider_tree_depth, n_nodes-1)  # Ensure depth does not exceed n_nodes-1
        # longest path can be of size n_nodes-1, so we set reasonable limit
        self.max_path_length = min(max_path_length, n_nodes-1)
        # At most n_nodes-2 conditioning variables
        self.max_conditioning_set_size = min(max_conditioning_set_size, n_nodes-2)

    def _add_assumption(self, name: str, initial_weight: float):
        if name not in self.name_to_assumption:
            self.name_to_sentence[name] = Assumption(name=name,
                                                     contrary=asm.contrary(name),
                                                     initial_weight=initial_weight)
            self.name_to_sentence[asm.contrary(name)] = Sentence(name=asm.contrary(name))
            self.name_to_assumption[name] = self.name_to_sentence[name]
        else:
            logger.warning(f"Assumption {name} already exists, skipping creation.")

    def _add_arr_and_mutual_exclusion_arguments(self):
        """Add arrow and no-edge assumptions for all pairs of nodes.
        Also add mutual exclusion arguments for arrows and no-edge assumptions for each pair of nodes.
        """
        for node1, node2 in combinations(range(self.n_nodes), 2):
            # Add arrow assumptions
            self._add_assumption(asm.arr(node1, node2), initial_weight=self.default_weight)
            self._add_assumption(asm.arr(node2, node1), initial_weight=self.default_weight)
            # Add no-edge assumption
            self._add_assumption(asm.noe(node1, node2), initial_weight=self.default_weight)

            # Add mutual exclusion arguments for arrows
            assums = [asm.arr(node1, node2), asm.arr(node2, node1), asm.noe(node1, node2)]
            for assum1 in assums:
                for assum2 in assums:
                    if assum1 != assum2:
                        arg = Argument(
                            claim=self.name_to_sentence[asm.contrary(assum2)],
                            premise=[self.name_to_sentence[assum1]]
                        )
                        self.arguments.add(arg)

    def _add_cycle_arguments(self):
        """Add cycle arguments for all cycles of size 3 to max_cycle_size.
        Cycles are added as arguments with the assumption that they are mutually exclusive.
        """
        if self.max_cycle_size < 3:
            logger.warning("Max cycle size is less than 3, skipping cycle arguments.")
            return

        # get all unique arrangements of nodes of size cycle-size
        # to get arrangements, first get all combinations, then consdider all permutations
        # each arrangement is considered a cycle, attaks arrow that goes from last node to first node
        for cycle_size in range(3, self.max_cycle_size + 1):
            for cycle_nodes in combinations(range(self.n_nodes), cycle_size):
                # Generate all permutations corresponding to all possible cycles
                for perm in permutations(cycle_nodes):
                    arg = Argument(
                        claim=self.name_to_sentence[asm.arr(perm[-1], perm[0])],
                        premise=[self.name_to_sentence[asm.arr(perm[i], perm[i+1])]
                                 for i in range(cycle_size - 1)]
                    )
                    self.arguments.add(arg)

    def _add_indep_assums_and_indep_noe_arguments(self):
        """Add independence assumptions and argument that independence implies no-edge.
        Independence assumptions are added for all pairs of nodes and all conditioning sets 
        up to self.max_conditioning_set_size.
        """
        if self.max_conditioning_set_size < 0:
            logger.error("Max conditioning set size is less than 0, exitting.")
            raise ValueError("Max conditioning set size must be non-negative.")

        for node1, node2 in combinations(range(self.n_nodes), 2):
            for conditioning_set_size in range(self.max_conditioning_set_size + 1):
                for conditioning_set in combinations(
                        set(range(self.n_nodes)) - {node1, node2},
                        conditioning_set_size):
                    # Add independence assumption
                    self._add_assumption(asm.indep(node1, node2, conditioning_set),
                                         initial_weight=self.default_weight)

                    # Add argument that independence implies no-edge
                    arg = Argument(
                        claim=self.name_to_sentence[asm.noe(node1, node2)],
                        premise=[self.name_to_sentence[asm.indep(node1, node2, conditioning_set)]]
                    )
                    self.arguments.add(arg)

    def _get_path_solutions(self, path_nodes, conditioning_set):
        """Generate all possible active collider trees on path_nodes.
        Uses a backtracking approach to find all valid combinations of arrows

        Rough sketch of the algorithm:
        1. if we reached the end of the path, terminate
        2. try adding one of the 2 candidate arrows
            - Is mutual exclusivity satisfied?, no: terminate, yes go to next check
            - If the current node is in the conditioning set
                - if is collider: good, move on to recursive call
                - if not collider: bad, terminate this solution
            - If the current node is not in the conditioning set
                - if not collider: good, move on to recursive call
                - if collider: branch out to all descendants that are in the conditioning set, up to certain distance
                    Each possible branch continues as separate recursive call.
        3. try adding the other candidate arrow.
        4. Finnish.
        """
        solutions = []

        def dfs(previous_arrow_on_path=None, current_node_id=0, arrows_so_far=set()):
            if current_node_id == len(path_nodes) - 1:
                # Base case of recursion: we reached the end of the path, so we can add the solution
                solutions.append((frozenset(arrows_so_far)))  # essentially just copying the arrows_so_far
                return
            else:
                # Recursive case: consider candidate arrows, see if they form a valid partial solution
                for candidate_arrow in [
                    (path_nodes[current_node_id], path_nodes[current_node_id + 1]),
                    (path_nodes[current_node_id + 1], path_nodes[current_node_id])
                ]:
                    reversed_candidate_arrow = (candidate_arrow[1], candidate_arrow[0])
                    if reversed_candidate_arrow in arrows_so_far:  # mutual exclusivity check
                        continue
                    elif previous_arrow_on_path is None:  # equivalent to current_node_id == 0:
                        # case 0: no previous arrow, just add the candidate arrow and continue solving
                        arrows_so_far.add(candidate_arrow)
                        dfs(previous_arrow_on_path=candidate_arrow,
                            current_node_id=current_node_id + 1,
                            arrows_so_far=arrows_so_far)
                        arrows_so_far.remove(candidate_arrow)
                    elif path_nodes[current_node_id] in conditioning_set:
                        # case 1: current node is in the conditioning set
                        if (previous_arrow_on_path[1] == path_nodes[current_node_id]
                                and candidate_arrow[1] == path_nodes[current_node_id]):
                            # collider case, good, add candidate arrow and continue solving
                            arrows_so_far.add(candidate_arrow)
                            dfs(previous_arrow_on_path=candidate_arrow,
                                current_node_id=current_node_id + 1,
                                arrows_so_far=arrows_so_far)
                            arrows_so_far.remove(candidate_arrow)  # backtrack
                        else:  # bad, terminate this solution, path is blocked
                            continue
                    else:
                        # case 2: current node is not in the conditioning set
                        if (previous_arrow_on_path[1] == path_nodes[current_node_id]
                                and candidate_arrow[1] == path_nodes[current_node_id]):  # collider case
                            # check out all possible descendants and paths to them that can help avoid blocking
                            parent = path_nodes[current_node_id]
                            branches = self._get_all_descendant_branches(parent, conditioning_set)
                            for branch in branches:  # check mutual exclusivity
                                if any((arr[1], arr[0]) in arrows_so_far for arr in branch):
                                    continue
                                diff_set = branch - arrows_so_far  # add branch arrows to the set
                                arrows_so_far.update(diff_set)
                                dfs(previous_arrow_on_path=candidate_arrow,  # continue solving with the branch arrows added
                                    current_node_id=current_node_id + 1,
                                    arrows_so_far=arrows_so_far)
                                arrows_so_far.difference_update(diff_set)  # backtrack, remove the branch arrows
                        else:  # not collider case, good, add and move on
                            arrows_so_far.add(candidate_arrow)
                            dfs(previous_arrow_on_path=candidate_arrow,
                                current_node_id=current_node_id + 1,
                                arrows_so_far=arrows_so_far)
                            arrows_so_far.remove(candidate_arrow)  # backtrack
        return solutions

    def _get_all_descendant_branches(self, parent, conditioning_set) -> List[Set[Tuple]]:
        if parent in conditioning_set:
            logger.error("When building collider trees: parent node is in the conditioning set, cannot find descendants.")
            raise ValueError("Parent node cannot be in the conditioning set.")

        if self.max_collider_tree_depth == 0:
            return []

        branches = []
        for descendant in conditioning_set:
            for num_intermediate_nodes in range(0, self.max_collider_tree_depth):
                for intermediate_nodes in combinations(
                        set(range(self.n_nodes)) - {parent, descendant},
                        num_intermediate_nodes):
                    for permutation in permutations(intermediate_nodes):
                        branch_nodes_sequence = (parent, *permutation, descendant)
                        branch_arrows = frozenset({
                            (branch_nodes_sequence[i], branch_nodes_sequence[i + 1])
                            for i in range(len(branch_nodes_sequence) - 1)
                        })
                        branches.append(branch_arrows)
        return branches

    def _add_active_path_assums_and_arguments(self):
        """Add active path assumptions and arguments.
        Active path assumptions are added for all pairs of nodes and all paths with length up to self.max_path.
        Z-activr collider tree arrows that support active paths are considered up to self.max_collider_tree_depth.
        Active path conditioning node sets are considered up to self.max_conditioning_set_size.
        
        Using the symmetry of nodes with each other.
        We can find one solutions for given pair of nodes and then use symmetry to find solutions for other nodes.
        Using symmetry would be basically swapping places in the paths, conditioning sets and active collider trees.
        """
        if self.max_path_length < 2:
            logger.error("Max path length is less than 2, cannot add active path assumptions.")
            raise ValueError("Max path length must be at least 2.")

        raise NotImplementedError("Active path assumptions and arguments are not implemented yet.")

    def build_arguments(self):
        """Build all arguments for the BSAF.
        This method should be called after all assumptions are added.
        """
        self._add_arr_and_mutual_exclusion_arguments()
        self._add_cycle_arguments()
        self._add_indep_assums_and_indep_noe_arguments()
        self._add_active_path_assums_and_arguments()

    def create_bsaf(self):
        """
        Create the BSAF with the assumptions and arguments built from the facts.
        """
        self.build_arguments()

        # Create BSAF with the assumptions and arguments
        self.bsaf = BSAF(assumptions=set(self.name_to_assumption.values()),
                         arguments=self.arguments)

        return self.bsaf
