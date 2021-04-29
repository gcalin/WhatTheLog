#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from typing import List, Union, Set, Dict, Tuple

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
    def __init__(self):
        self.states: Dict[State, Tuple[Dict[State, Edge], Dict[State, Edge]]] = {}
        self.edges: Set = set()

    def add_state(self, state: State):
        """
        Method to add a new state to the graph.

        :param state: State to add
        :raises StateAlreadyExistsException: if state already exists
        """

        if state in self:
            raise StateAlreadyExistsException()

        self.states[state] = ({}, {})

    def add_edge(self, edge: Edge) -> bool:
        """
        Method to add an edge to the graph

        :param edge: Edge to be added
        :return: Whether adding the edge was successful. If one of the nodes in
        edge does not exist or edge already exists returns False else True.
        """
        start = edge.start
        end = edge.end
        if start not in self:
            return False
        elif end not in self:
            return False
        elif end in self.states[start][1]:
            return False
        else:
            self.states[start][1][end] = edge
            self.states[end][0][start] = edge
            self.edges.add(edge)
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
            return list(self.states[state][1].values())
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
            return list(self.states[state][1].keys())
        else:
            return None

    def get_incoming_edges(self, state: State) -> Union[List[Edge], None]:
        """
        Method to get incoming edges of a state.

        :param state: State to get incoming edges for
        :return: List of incoming edges from state.
        If state does not exist return None.
        """
        if state in self:
            return list(self.states[state][0].values())
        else:
            return None

    def get_incoming_states(self, state: State) -> Union[List[State], None]:
        """
        Method to get incoming states of a state.

        :param state: State to get incoming states for
        :return: List of incoming edges from state.
        If state does not exist return None.
        """
        if state in self:
            return list(self.states[state][0].keys())
        else:
            return None

    def __str__(self):
        return str(self.states)

    def __contains__(self, item: State):
        return item in self.states
