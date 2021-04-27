from whatthelog.prefixtree.state import State


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
