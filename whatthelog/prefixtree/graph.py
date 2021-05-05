#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from typing import List, Union, Dict

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.state import State
from whatthelog.auto_printer import AutoPrinter
from whatthelog.exceptions import StateAlreadyExistsException
from whatthelog.prefixtree.sparse_matrix import SparseMatrix
from whatthelog.prefixtree.edge_properties import EdgeProperties


#****************************************************************************************************
# Graph
#****************************************************************************************************

class Graph(AutoPrinter):
    """
    Class implementing a graph
    """

    __slots__ = ['edges', 'states', 'state_indices_by_hash', 'states_by_prop']

    def __init__(self):
        self.edges = SparseMatrix()
        self.states: List[State] = []
        self.state_indices_by_hash: Dict[int, int] = {}
        self.states_by_prop: Dict[int, int] = {}

    def get_state_by_hash(self, state_hash: int):
        """
        Method to fetch the state object from its hash.
        :param state_hash: the hash of the state to fetch
        :return: the state object
        """
        return self.states[self.state_indices_by_hash[state_hash]]

    def add_state(self, state: State):
        """
        Method to add a new state to the graph.

        :param state: State to add
        :raises StateAlreadyExistsException: if state already exists
        """

        if state in self:
            raise StateAlreadyExistsException()

        curr_index = len(self.states)
        self.states.append(state)
        self.state_indices_by_hash[hash(state)] = curr_index

        if state.properties.get_prop_hash() in self.states_by_prop:
            state.properties = self.get_state_by_hash(
                self.states_by_prop[state.properties.get_prop_hash()]).properties
        else:
            self.states_by_prop[state.properties.get_prop_hash()] = hash(state)

    def add_edge(self, start: State, end: State, props: EdgeProperties) -> bool:
        """
        Method to add an edge to the graph

        :param start: Origin state of the edge
        :param end: Destination state of the edge
        :param props: the edge properties object
        :return: Whether adding the edge was successful. If one of the nodes in
        edge does not exist or edge already exists returns False else True.
        """

        if hash(start) not in self.state_indices_by_hash or hash(end) not in self.state_indices_by_hash:
            return False

        start_index = self.state_indices_by_hash[hash(start)]
        end_index = self.state_indices_by_hash[hash(end)]
        if not (start_index, end_index) in self.edges:
            self.edges[start_index, end_index] = str(props)
            return True
        return False

    def size(self):
        """
        Method to get the size of the graph.

        :return: Number of states
        """
        return len(self.states)

    def get_outgoing_props(self, state: State) -> Union[List[EdgeProperties], None]:
        """
        Method to get outgoing edges of a state.

        :param state: State to get outgoing edges for
        :return: List of outgoing edges from state.
        If state does not exist return None.
        """
        if state in self:
            results = self.edges.find_children(self.state_indices_by_hash[hash(state)])
            return [EdgeProperties.parse(result[1]) for result in results]
        else:
            return None

    def get_outgoing_states(self, state: State) -> Union[List[State], None]:
        """
        Method to get outgoing states of a state.

        :param state: State to get outgoing states for
        :return: List of outgoing edges from state.
        If state does not exist return None.
        """
        if state in self:
            results = self.edges.find_children(self.state_indices_by_hash[hash(state)])
            return [self.states[result[0]] for result in results] if results else []
        else:
            return None

    def __str__(self):
        return str(self.states)

    def __contains__(self, item: State):
        return hash(item) in self.state_indices_by_hash

    def __len__(self):
        return len(self.states)
