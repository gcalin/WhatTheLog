from whatthelog.prefixtree.graph import Graph
import abc


class Search(abc.ABC):
    def __init__(self, initial_solution: Graph):
        self.solution = initial_solution

    @abc.abstractmethod
    def search(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_neighborhood(self, state: Graph, *args, **kwargs):
        pass

    @abc.abstractmethod
    def random_neighbor(self, state: Graph, *args, **kwargs):
        pass


class SimulatedAnnealing(Search):
    def get_neighborhood(self, state: Graph, *args, **kwargs):
        pass

    def random_neighbor(self, state: Graph, *args, **kwargs):
        pass

    def search(self, *args, **kwargs):
        m: int = 0
