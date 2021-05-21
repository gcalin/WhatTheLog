import random
from copy import deepcopy
from typing import List

from numpy import exp, stddev, log2

from whatthelog.prefixtree.evaluator import Evaluator
from whatthelog.prefixtree.graph import Graph
import abc

from whatthelog.simulatedannealing.action import MergeRandomState
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


class Search(abc.ABC):
    def __init__(self,
                 initial_solution: Graph,
                 syntax_tree: SyntaxTree,
                 positive_traces_dir: str,
                 negative_traces_dir: str):
        self.solution = initial_solution
        self.syntax_tree = syntax_tree
        self.positive_traces_dir = positive_traces_dir
        self.negative_traces_dir = negative_traces_dir

    @abc.abstractmethod
    def search(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def random_neighbor(self, state: Graph, *args, **kwargs):
        pass


class SimulatedAnnealing(Search):
    def __init__(self, initial_solution: Graph,
                 syntax_tree: SyntaxTree,
                 positive_traces_dir: str,
                 negative_traces_dir: str
                 ):
        super().__init__(initial_solution, syntax_tree,
                         positive_traces_dir, negative_traces_dir)

    def random_neighbor(self, state: Graph = None, *args, **kwargs) -> Graph:
        copy = deepcopy(state)
        self.__random_action().perform(copy)
        return copy

    @staticmethod
    def __random_action():
        return random.choice([MergeRandomState])

    def neighborhood_size(self):
        # The number of edges defines how many nodes can be merged with their child
        return len(self.solution.edges)

    def chain_length_aarts(self):
        return self.neighborhood_size()

    # def chain_length_trivial(self, k: int):

    def update_temp_aarts(self, t: float, delta: float, sigma: float):
        return t / (1 + t*log2(1+delta)/3*sigma)

    def search(self, *args, **kwargs):

        k_max: int = 100

        # Initial temperature
        t: int = 1

        # Inner iteration counter
        m: int = 0

        # Outer iteration counter
        k: int = 0

        eval: Evaluator = Evaluator(self.solution, self.syntax_tree,
                                    self.positive_traces_dir, self.negative_traces_dir)

        number_of_steps: int = self.chain_length_aarts()

        while k < k_max:
            vals: List[float] = []
            while m < number_of_steps:
                neighbor: Graph = self.random_neighbor()
                other_eval: Evaluator = Evaluator(neighbor, self.syntax_tree,
                                                  self.positive_traces_dir, self.negative_traces_dir)
                e1 = eval.evaluate()
                e2 = other_eval.evaluate()
                d: float = e1 - e2

                if d < 0 or (random.random() < exp(d/t)):
                    eval.update(neighbor)
                    self.solution = neighbor

                vals.append(e2)
                m += 1
            t = self.update_temp_aarts(t, 0.1, stddev(vals))
