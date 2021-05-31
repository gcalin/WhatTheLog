import os
import random
from copy import deepcopy
from pathlib import Path
from typing import List

from numpy import exp

from scripts.cross_validation import train_test_validation_split
from whatthelog.prefixtree.evaluator import Evaluator
from whatthelog.prefixtree.graph import Graph
import abc

from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.simulatedannealing.action import MergeRandomState
from whatthelog.simulatedannealing.coolingschedule import CoolingSchedule, BonomiLuttonSchedule
from whatthelog.simulatedannealing.selection import Selection, RandomSelection
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory


class Search(abc.ABC):
    def __init__(self,
                 initial_solution: Graph,
                 syntax_tree: SyntaxTree,
                 positive_traces_dir: str,
                 negative_traces_dir: str,
                 selection_method: Selection,
                 cooling_schedule: CoolingSchedule):
        self.solution = initial_solution
        self.syntax_tree = syntax_tree
        self.positive_traces_dir = positive_traces_dir
        self.negative_traces_dir = negative_traces_dir
        self.selection = selection_method
        self.cooling_schedule = cooling_schedule

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
                 negative_traces_dir: str,
                 selection_method: Selection,
                 cooling_schedule: CoolingSchedule):
        super().__init__(initial_solution, syntax_tree,
                         positive_traces_dir, negative_traces_dir,
                         selection_method, cooling_schedule)

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

    def search(self, *args, **kwargs):

        k_max: int = 1

        # Initial temperature
        t: float = self.cooling_schedule.initial_temperature

        # Outer iteration counter
        k: int = 0

        evaluator: Evaluator = Evaluator(self.solution, self.syntax_tree,
                                         self.positive_traces_dir, self.negative_traces_dir)



        while k < k_max:
            vals: List[float] = []

            # Inner iteration counter
            m = 0
            number_of_steps: int = self.cooling_schedule.chain_length

            while m < number_of_steps:
                neighbour: Graph = deepcopy(self.solution)
                self.selection.update(self.solution)
                s1, s2 = self.selection.select()
                neighbour.full_merge_states(s1, s2)

                other_eval: Evaluator = Evaluator(neighbour, self.syntax_tree,
                                                  self.positive_traces_dir, self.negative_traces_dir)
                e1 = evaluator.evaluate()
                e2 = other_eval.evaluate()
                d: float = e1 - e2

                if d < 0 or (random.random() < exp(d/t)):
                    evaluator.update(neighbour)
                    self.solution = neighbour

                vals.append(e2)
                m += 1
            t = self.cooling_schedule.update_temperature()


if __name__ == '__main__':

    project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
    cfg_file = str(project_root.joinpath('resources/config.json'))
    true_traces = str(project_root.joinpath('resources/logs_train'))
    val_dir_t = str(project_root.joinpath('resources/logs_validation_true'))
    val_dir_f = str(project_root.joinpath('resources/logs_validation_false'))
    model = PrefixTreeFactory.get_prefix_tree(true_traces, cfg_file, True)
    s_tree = SyntaxTreeFactory().parse_file(cfg_file)
    # train_dir, val_dir_t, val_dir_f, test_dir_t, test_dir_f, model =\
    #     train_test_validation_split(s_tree,
    #                                 true_traces,
    #                                 0.25,
    #                                 0.25,
    #                                 cfg_file,
    #                                 True)

    sa = SimulatedAnnealing(model, s_tree, val_dir_t, val_dir_f, RandomSelection(model), BonomiLuttonSchedule())
    sa.search()
