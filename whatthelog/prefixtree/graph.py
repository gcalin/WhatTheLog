from typing import List, Dict, Union

from whatthelog.prefixtree.edge import Edge
from whatthelog.prefixtree.state import State


class Graph:
    """
    Class implementing a graph
    """
    def __init__(self):
        self.states: Dict[State, Dict[State, Edge]] = {}

    def add_state(self, state: State):
        """
        Method to add a new state to the graph

        :param state:
        """
        self.states[state] = {}

    def add_edge(self, edge: Edge) -> bool:
        """
        Method to add an edge to the graph

        :param edge: Edge to be added
        :return: Whether adding the edge was successful. If one of the nodes in
        edge does not exist or edge already exists returns False else True.
        """
        start = edge.start
        end = edge.end
        if start not in self.states:
            return False
        elif end in self.states[start]:
            return False
        else:
            self.states[edge.start][edge.end] = edge
            return True

    def size(self):
        """
        Method to get the size of the graph.

        :return: Number of states
        """
        return len(self.states)

    def get_edges(self):
        """
        Method to get all the edges in the graph.

        :return: A list of all edges
        """
        edges = []
        for edgesMap in self.states.values():
            edges += edgesMap.values()
        return edges

    def get_outgoing_edges(self, state: State) -> Union[List[Edge], None]:
        """
        Method to get outgoing edges of a state.

        :param state: State to get outgoing edges for
        :return: List of outgoing edges from state.
        If state does not exist return None.
        """
        if state in self.states:
            return list(self.states[state].values())
        else:
            return None

    def get_outgoing_states(self, state: State) -> Union[List[State], None]:
        """
        Method to get outgoing states of a state.

        :param state: State to get outgoing states for
        :return: List of outgoing edges from state.
        If state does not exist return None.
        """
        if state in self.states:
            return list(self.states[state].keys())
        else:
            return None

    def __str__(self):
        return str(self.states)
