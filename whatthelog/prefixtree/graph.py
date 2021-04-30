#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from typing import List, Union, Dict
from array import array

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.state import State
from whatthelog.prefixtree.edge import Edge
from whatthelog.auto_printer import AutoPrinter
from whatthelog.exceptions import StateAlreadyExistsException


#****************************************************************************************************
# Graph
#****************************************************************************************************

class Graph(AutoPrinter):
    """
    Class implementing a graph
    """

    __slots__ = ['states', 'edges', 'state_indices_by_hash', 'states_by_prop', 'children']

    def __init__(self):
        self.states: List[State] = []
        self.edges: List[Edge] = []
        self.state_indices_by_hash: Dict[int, int] = {}
        self.states_by_prop: Dict[str, int] = {}
        self.children: Dict[int, array[int]] = {}

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

    def add_edge(self, state: State, edge: Edge) -> bool:
        """
        Method to add an edge to the graph

        :param state: Origin state of the edge
        :param edge: Edge to be added
        :return: Whether adding the edge was successful. If one of the nodes in
        edge does not exist or edge already exists returns False else True.
        """
        start = hash(state)
        end = edge.end
        if start not in self.state_indices_by_hash:
            return False
        elif end not in self.state_indices_by_hash:
            return False
        elif start in self.children and end in \
                [self.edges[index].end for index in self.children[start]]:
            return False
        else:
            curr_edge = len(self.edges)
            self.edges.append(edge)
            if start in self.children:
                self.children[start].append(curr_edge)
            else:
                self.children[start] = array('l', [curr_edge])
        return True

    def size(self):
        """
        Method to get the size of the graph.

        :return: Number of states
        """
        return len(self.states)

    def get_outgoing_edges(self, state: State) -> Union[List[Edge], None]:
        """
        Method to get outgoing edges of a state.

        :param state: State to get outgoing edges for
        :return: List of outgoing edges from state.
        If state does not exist return None.
        """
        if state in self:
            if hash(state) in self.children:
                return [self.edges[index] for index in self.children[hash(state)]]
            else:
                return []
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
            if hash(state) in self.children:
                return [self.get_state_by_hash(edge.end) for edge in
                        [self.edges[index] for index in self.children[hash(state)]]]
            else:
                return []
        else:
            return None

    # def get_incoming_edges(self, state: State) -> Union[List[Edge], None]:
    #     """
    #     Method to get incoming edges of a state.
    #
    #     :param state: State to get incoming edges for
    #     :return: List of incoming edges from state.
    #     If state does not exist return None.
    #     """
    #     if state in self:
    #         return list(self.states[state][0])
    #     else:
    #         return None

    # def get_incoming_states(self, state: State) -> Union[List[State], None]:
    #     """
    #     Method to get incoming states of a state.
    #
    #     :param state: State to get incoming states for
    #     :return: List of incoming edges from state.
    #     If state does not exist return None.
    #     """
    #     if state in self:
    #         return list([edge.end for edge in self.states[state][0]])
    #     else:
    #         return None

    def __str__(self):
        return str(self.states)

    def __contains__(self, item: State):
        return hash(item) in self.state_indices_by_hash
