from typing import List, Dict


class State:
    """
    Class representing a state. Holds a list of all log template ids
    represented by this state.
    """
    def __init__(self, log_ids: List[int]):
        """
        State constructor.

        :param log_ids: The log template ids this state holds.
        """
        self.log_ids: List[int] = log_ids
        self.outgoing: Dict['State', Edge] = {}
        self.incoming: Dict['State', Edge] = {}

    def __str__(self):
        if len(self.log_ids) == 1:
            return str(self.log_ids[0])

        return str(self.log_ids)

    def __repr__(self):
        return self.__str__()


class Edge:
    """
    Class representing an edge in a graph.
    """
    def __init__(self, start: State, end: State):
        """
        Edge constructor.

        :param start: Start state.
        :param end: End state.
        """
        self.start = start
        self.end = end

    def __str__(self):
        return str(self.start) + " -> " + str(self.end)

    def __repr__(self):
        return self.__str__()
