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

    def set_root(self, state: State):
        """
        Root setter.

        :param state: State to set as tree's root
        """
        self.__root = state

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
        self.add_state(state)
        self.add_edge(Edge(parent, state))

    def get_parent(self, state: State) -> Union[State, None]:
        """
        Method to get the parent of a state.

        :param state: State to get parent of
        :return: Parent of state. If None state is the root.
        """
        parent = self.get_incoming_states(state)
        if parent is None or len(parent) == 0:
            return None
        else:
            return parent[0]


class InvalidTreeException(Exception):
    pass
