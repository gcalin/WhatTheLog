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
from whatthelog.prefixtree.adjacency_graph import AdjacencyGraph
from whatthelog.prefixtree.state import State
from whatthelog.exceptions import InvalidTreeException


#****************************************************************************************************
# Prefix Tree
#****************************************************************************************************

class PrefixTree(AdjacencyGraph):
    """
    Prefix tree implemented using an adjacency map-based graph.
    """

    __slots__ = ['syntax_tree', 'states', 'state_indices_by_id', 'prop_by_hash',
                 'start_node', 'outgoing_edges', 'incoming_edges']

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
        parents = self.get_incoming_states(state)
        assert len(parents) <= 1, "Edge has more than one parent!"

        if parents is None or len(parents) == 0:
            return None
        else:
            return parents[0]

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
