#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import annotations
from typing import List, Union, Tuple

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.prefixtree.edge_properties import EdgeProperties
from whatthelog.prefixtree.graph import Graph
from whatthelog.prefixtree.state import State
from whatthelog.exceptions import InvalidTreeException


#****************************************************************************************************
# Prefix Tree
#****************************************************************************************************

class PrefixTree(Graph):
    """
    Prefix tree implemented using an adjacency map-based graph.
    """

    def __init__(self, syntax_tree: SyntaxTree, root: State):
        super().__init__(syntax_tree, root)

    def get_root(self) -> State:
        """
        Root getter.

        :return: the root of the tree
        """
        return self.start_node

    def get_children(self, state: State) -> List[State]:
        """
        Method to get children of a state.

        :param state: State to get children of
        :return: List of children. If empty this state is a leaf.
        """
        return self.get_outgoing_states(state)

    def add_child(self, state: State, parent: State, props: EdgeProperties = EdgeProperties()):
        """
        Method to add a child in the tree.
        Requires that the parent be in the current tree.

        :param state: State to add
        :param parent: Parent of state to link with
        :param props: the edge properties object
        """

        assert parent in self, "Parent is not in the tree!"

        self.add_state(state)
        self.add_edge(parent, state, props)

    def add_branch(self, state: State, tree: PrefixTree, parent: State):
        """
        Appends a branch from the input tree starting at the given state
        to the current tree under the given parent.
        Requires that the parent be in the current tree,
        and that the branch must not contain nodes already in the current tree.
        :param state: the root of the branch to add
        :param tree: the tree where the branch originates from
        :param parent: the node in the current tree to append the branch to
        """

        assert parent in self, "Parent is not in the tree!"

        queue = [(state, parent)]
        while queue:

            current, parent = queue.pop(0)
            assert current not in self, "Branch state is already in current tree!"

            self.add_child(current, parent)

            for child in tree.get_children(current):
                queue.append((child, current))

    def get_parent(self, state: State) -> Union[State, None]:
        """
        Method to get the parent of a state.
        WARNING: This method is O(n) were n is the number of edges in the tree!

        :param state: State to get parent of
        :return: Parent of state. If None state is the root.
        """

        assert state in self
        parents = self.edges.get_parents(self.state_indices_by_id[id(state)])
        assert len(parents) <= 1, "Edge has more than one parent!"

        if parents is None or len(parents) == 0:
            return None
        else:
            return self.states[parents[0]]

    def merge(self, other: PrefixTree):
        """
        Merges another tree into the current one.
        The tree's children are appended to this one's, the tree's parent is discarded.
        Requires the input tree to have the same root as this one.
        Assumes the tree is coherent: there are no duplicated children in any node.

        :param other: the tree to be merged into this one.
        """

        if not self.start_node.is_equivalent(other.get_root()):
            raise InvalidTreeException("Merge failed: source tree does not have same root as destination tree!")

        stack = [(self.start_node, other.get_root())]
        while True:

            this_state, that_state = stack.pop()

            for that_child in other.get_children(that_state):
                conflict = False
                for this_child in self.get_children(this_state):
                    if that_child.is_equivalent(this_child):
                        stack.append((this_child, that_child))
                        conflict = True

                if not conflict:
                    self.add_branch(that_child, other, this_state)

            if not stack:
                break

    def get_adj_list(self, remove_self_loops: bool = False) -> List[Tuple[int, int, float]]:
        """
        Returns an adjacency list representing the prefix tree, as a list of tuples in the form:
            (edge_start, edge_end, edge_weight).

        Each edge's weight is the frequency with which that edge is used from its parent,
        in other words: if node A has two children B and C, with edge AB having 6 passes
        and edge AC having 4 passes, then edge AB will have weight 0.6 andAC 0.4.

        :param remove_self_loops: if True the output will not contain self-looping edges
                                  (edges with same origin as destination)
        :return: the adjacency matrix for the prefix tree
        """

        return self.edges.get_weights_list(remove_self_loops)


#****************************************************************************************************
# Prefix Tree Iterator
#****************************************************************************************************

class TreeIterator:
    """
    Iterator class for the tree.
    Return states in a Breadth-First Search.
    """

    def __init__(self, tree: PrefixTree):
        self.tree = tree
        self.queue = [tree.get_root()]

    def __next__(self):

        if not self.queue:
            raise StopIteration

        current = self.queue.pop(0)
        for child in self.tree.get_children(current):
            self.queue.append(child)

        return current
