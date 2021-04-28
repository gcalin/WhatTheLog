from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union

from whatthelog.prefixtree.graph import Graph
from whatthelog.prefixtree.state import State, Edge


class PrefixTreeGraph(Graph):
    """
    Prefix tree implemented using a adjacency map graph
    """

    def __init__(self, root: State):
        super().__init__()
        self.__root = root
        self.add_state(root)

    def get_root(self) -> State:
        """
        Root getter.

        :return: the root of the tree
        """
        return self.__root

    def get_children(self, state: State) -> List[State]:
        """
        Method to get children of a state.

        :param state: State to get children of
        :return: List of children. If empty this state is a leaf.
        """
        return self.get_outgoing_states(state)

    def add_child(self, state: State, parent: State):
        """
        Method to add a child in the tree.

        :param state: State to add
        :param parent: Parent of state to link with
        """

        assert parent in self.states, "Parent is not in the tree!"

        state.incoming = {}
        self.add_state(state)
        self.add_edge(Edge(parent, state))

        queue = list(state.outgoing.keys())
        while queue:

            child = queue.pop(0)
            self.add_state(child)
            queue += child.outgoing.keys()

    def get_parent(self, state: State) -> Union[State, None]:
        """
        Method to get the parent of a state.

        :param state: State to get parent of
        :return: Parent of state. If None state is the root.
        """

        parents = self.get_incoming_states(state)
        assert len(parents) <= 1, "Edge has more than one parent!"

        if parents is None or len(parents) == 0:
            return None
        else:
            return parents[0]


class TreeIterator:
    """
    Iterator class for the tree.
    Return states in a Breadth-First Search.
    """

    def __init__(self, tree: PrefixTreeGraph):
        self.tree = tree
        self.queue = [tree.get_root()]

    def __next__(self):

        if not self.queue:
            raise StopIteration

        current = self.queue.pop(0)
        for child in list(current.outgoing.keys()):
            self.queue.append(child)

        return current

@dataclass(frozen=True)
class InvalidTreeException(Exception):
    message: str = field(default="Tree is invalid")
