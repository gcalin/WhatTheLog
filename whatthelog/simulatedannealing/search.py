import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union, Dict

import numpy as np
from numpy import exp, std, mean, array_split

from scripts.log_scrambler import produce_false_trace
from whatthelog.prefixtree.evaluator import Evaluator
from whatthelog.prefixtree.graph import Graph
import abc

from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.simulatedannealing.action import MergeRandomState
from whatthelog.simulatedannealing.coolingschedule import CoolingSchedule, LundySchedule
from whatthelog.simulatedannealing.selection import Selection, TournamentSelection
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


class RandomSearch(Search):
    def __init__(self, initial_solution: Graph,
                 syntax_tree: SyntaxTree,
                 positive_traces_dir: str,
                 negative_traces_dir: str):
        super().__init__(initial_solution, syntax_tree,
                         positive_traces_dir, negative_traces_dir,
                         selection_method=None, cooling_schedule=None)

    def select_random_states(self, model: Graph):
        return model.get_non_terminal_states()

    def search(self,
               max_iterations: int = 100,
               debug=False,
               *args, **kwargs):

        evaluator: Evaluator = Evaluator(None, self.syntax_tree,
                                         self.positive_traces_dir,
                                         self.negative_traces_dir,
                                         weight_size=0.33,
                                         weight_accuracy=0.66,
                                         initial_size=len(self.solution))

        if debug:
            print("Parsing true validation traces...")

        true_traces: List[PrefixTree] = PrefixTreeFactory.\
            parse_multiple_traces(self.positive_traces_dir, self.syntax_tree, True)

        if debug:
            print("Parsing false validation traces...")

        false_traces: List[PrefixTree] = PrefixTreeFactory. \
            parse_multiple_traces(self.negative_traces_dir, self.syntax_tree, True)

        evaluator.update(self.solution)
        e, _,  _, _ = \
            evaluator.evaluate(true_trees=true_traces, false_trees=false_traces)

        for iteration in range(min(max_iterations, evaluator.initial_model_size)):
            if debug:
                print(f"Iteration {iteration}, size: {len(self.solution)}, eval: {e}")
            neighbour: Graph = deepcopy(self.solution)
            s1, s2 = self.select_random_states(neighbour)
            neighbour.merge_states(s1, s2, bool(random.getrandbits(1)))

            evaluator.update(neighbour)
            e1, _, _, _ = \
                evaluator.evaluate(true_trees=true_traces, false_trees=false_traces)

            if e1 < e:
                self.solution = neighbour
                e = e1

        return self.solution


class TournamentLocalSearch(Search):
    def __init__(self, initial_solution: Graph,
                 syntax_tree: SyntaxTree,
                 positive_traces_dir: str,
                 negative_traces_dir: str):
        super().__init__(initial_solution, syntax_tree,
                         positive_traces_dir, negative_traces_dir,
                         selection_method=TournamentSelection(initial_solution, 5, 0.75),
                         cooling_schedule=None)

    def search(self,
               max_iterations: int = 100,
               debug=False,
               *args, **kwargs):

        evaluator: Evaluator = Evaluator(None, self.syntax_tree,
                                         self.positive_traces_dir,
                                         self.negative_traces_dir,
                                         weight_size=0.33,
                                         weight_accuracy=0.66,
                                         initial_size=len(self.solution))

        if debug:
            print("Parsing true validation traces...")

        true_traces: List[PrefixTree] = PrefixTreeFactory.\
            parse_multiple_traces(self.positive_traces_dir, self.syntax_tree, True)

        if debug:
            print("Parsing false validation traces...")

        false_traces: List[PrefixTree] = PrefixTreeFactory. \
            parse_multiple_traces(self.negative_traces_dir, self.syntax_tree, True)

        evaluator.update(self.solution)
        e, _,  _, _ = \
            evaluator.evaluate(true_trees=true_traces, false_trees=false_traces)

        for iteration in range(min(max_iterations, evaluator.initial_model_size)):
            if debug:
                print(f"Iteration {iteration}, size: {len(self.solution)}, eval: {e}")
            neighbour: Graph = deepcopy(self.solution)
            self.selection.model = neighbour
            s1, s2 = self.selection.select()
            neighbour.merge_states(s1, s2, bool(random.getrandbits(1)))

            evaluator.update(neighbour)
            e1, _, _, _ = \
                evaluator.evaluate(true_trees=true_traces, false_trees=false_traces)

            if e1 < e:
                self.solution = neighbour
                e = e1

        return self.solution


class ParetoRandomSearch(Search):
    def __init__(self, initial_solution: Graph,
                 syntax_tree: SyntaxTree,
                 positive_traces_dir: str,
                 negative_traces_dir: str):
        super().__init__(initial_solution, syntax_tree,
                         positive_traces_dir, negative_traces_dir,
                         selection_method=None, cooling_schedule=None)

    def select_random_states(self, model: Graph):
        return model.get_non_terminal_states()

    def initial_set(self, how_many: int) -> List[Graph]:
        res: List[Graph] = []

        for _ in range(how_many):
            y = deepcopy(self.solution)

            # Update the selection model and select two states to merge
            s1, s2 = self.select_random_states(y)

            # Perform the merge
            y.merge_states(s1, s2, bool(random.getrandbits(1)))

            res.append(y)

        return res

    @staticmethod
    def dominates(f: Tuple[float, float, float], s: Tuple[float, float, float]) -> bool:
        """
        Return true if f pareto-dominates s.
        """
        return f[0] <= s[0] and f[1] <= s[1] and f[2] <= s[2]

    @staticmethod
    def remove_dominated_solutions(solution_set: List[Graph],
                                   evaluations: List[Tuple[float, float, float]],
                                   new_solution: Graph,
                                   new_evaluation: Tuple[float, float, float],
                                   debug: bool = False) \
            -> Tuple[List[Graph], List[Tuple[float, float, float]]]:

        # Get list of dominated solutions
        dominated: List[int] = []
        for count, evaluation in enumerate(evaluations):
            if SimulatedAnnealing.dominates(new_evaluation, evaluation):
                dominated.append(count)

        # If any dominated solutions found
        if dominated:
            # Remove the dominated solutions from the solution set
            for index in reversed(dominated):
                if debug:
                    print(f"Solution dominated with eval {evaluations[index]}")
                solution_set.pop(index)
                evaluations.pop(index)

            # Add the better solution to the set
            if debug:
                print(f"Better solution with eval {new_evaluation}")
            evaluations.append(new_evaluation)
            solution_set.append(new_solution)
        else:
            is_dominated: bool = False
            for evaluation in evaluations:
                if SimulatedAnnealing.dominates(evaluation, new_evaluation):
                    is_dominated = True
                    break
            if not is_dominated:
                solution_set.append(new_solution)
                evaluations.append(new_evaluation)
                if debug:
                    print(f"{new_evaluation} does not dominate anything, but was added to the solution set.")

        return solution_set, evaluations

    def search_mo(self,
                  front_size: int,
                  debug: bool = False,
                  *args, **kwargs):
        M: List[Graph] = []
        S: List[Graph] = self.initial_set(front_size)

        if debug:
            print(f"Sizes: {len(self.solution)}, {list(map(lambda x: len(x), S))}")

        evaluations_front: List[Tuple[float, float, float]] = []
        evaluations_solutions: List[Tuple[float, float, float]] = []
        evaluator: Evaluator = Evaluator(None, self.syntax_tree,
                                         self.positive_traces_dir,
                                         self.negative_traces_dir,
                                         initial_size=len(self.solution))

        # Outer iteration counter
        k: int = 0

        if debug:
            print("Parsing true validation traces...")

        true_traces: List[PrefixTree] = PrefixTreeFactory.\
            parse_multiple_traces(self.positive_traces_dir, self.syntax_tree, True)

        if debug:
            print("Parsing false validation traces...")

        false_traces: List[PrefixTree] = PrefixTreeFactory. \
            parse_multiple_traces(self.negative_traces_dir, self.syntax_tree, True)

        # For each initial solution
        for x in S:
            M.append(x)  # Add the solution to the potential efficient set
            evaluator.update(x)  # Evaluate the initial solution
            e2, sz, sp, re = evaluator.evaluate(true_trees=true_traces, false_trees=false_traces)
            evaluations_front.append((sz, sp, re))  # Add evaluation to the front
            evaluations_solutions.append((sz, sp, re))  # Add evaluation to the efficient set

        max_iterations = len(self.solution)

        while k < max_iterations:
            if debug:
                print(f"For iteration {k}, Solutions length: {len(self.solution)}")

            new_S: List[Graph] = []
            new_evaluations_front: List[Tuple[float, float, float]] = []

            for count, (x, evaluation) in enumerate(zip(S, evaluations_front)):
                # Create a copy of the current solution
                y: Graph = deepcopy(x)

                # Don't perform merges on the minimal state machine
                if len(y) > 3:
                    s1, s2 = self.select_random_states(y)

                    # Perform the merge
                    y.merge_states(s1, s2, bool(random.getrandbits(1)))

                # Update the evaluator
                evaluator.update(y)

                # Evaluate the model and calculate the difference
                e2, sz, sp, re = evaluator.evaluate(true_trees=true_traces, false_trees=false_traces)
                new_evaluation: Tuple[float, float, float] = (sz, sp, re)

                if not self.dominates(evaluation, new_evaluation):
                    self.remove_dominated_solutions(M, evaluations_solutions,
                                                    y, new_evaluation, debug=debug)

                # Accept a better solution with probability 1
                # And a worse solution with probability e^(d/t)
                if self.dominates(new_evaluation, evaluation):
                    # Update the next iteration front
                    new_S.append(y)
                    new_evaluations_front.append(new_evaluation)
                else:
                    # Update the next iteration front
                    new_S.append(x)
                    new_evaluations_front.append(evaluation)

            S = new_S
            evaluations_front = new_evaluations_front

            # Increment the iteration counter
            k += 1

        return M

    def search(self,
               max_iterations: int = 100,
               debug=False,
               *args, **kwargs):

        evaluator: Evaluator = Evaluator(None, self.syntax_tree,
                                         self.positive_traces_dir,
                                         self.negative_traces_dir,
                                         weight_size=0.33,
                                         weight_accuracy=0.66,
                                         initial_size=len(self.solution))

        if debug:
            print("Parsing true validation traces...")

        true_traces: List[PrefixTree] = PrefixTreeFactory.\
            parse_multiple_traces(self.positive_traces_dir, self.syntax_tree, True)

        if debug:
            print("Parsing false validation traces...")

        false_traces: List[PrefixTree] = PrefixTreeFactory. \
            parse_multiple_traces(self.negative_traces_dir, self.syntax_tree, True)

        evaluator.update(self.solution)
        e, _,  _, _ = \
            evaluator.evaluate(true_trees=true_traces, false_trees=false_traces)

        for iteration in range(min(max_iterations, evaluator.initial_model_size)):
            if debug:
                print(f"Iteration {iteration}, size: {len(self.solution)}, eval: {e}")
            neighbour: Graph = deepcopy(self.solution)
            s1, s2 = self.select_random_states(neighbour)
            neighbour.merge_states(s1, s2, bool(random.getrandbits(1)))

            evaluator.update(neighbour)
            e1, _, _, _ = \
                evaluator.evaluate(true_trees=true_traces, false_trees=false_traces)

            if e1 < e:
                self.solution = neighbour
                e = e1

        return self.solution


class SimulatedAnnealing(Search):
    def __init__(self, initial_solution: Graph,
                 syntax_tree: SyntaxTree,
                 positive_traces_dir: str,
                 negative_traces_dir: str,
                 selection_method: Selection,
                 cooling_schedule: CoolingSchedule,
                 output_file: str = "SA_output_8.txt"):
        super().__init__(initial_solution, syntax_tree,
                         positive_traces_dir, negative_traces_dir,
                         selection_method, cooling_schedule)
        self.output_file = output_file

    @staticmethod
    def __random_action():
        return random.choice([MergeRandomState])

    def search(self,
               debug: bool = False,
               max_iterations: int = 11,
               alpha: float = 1.05,
               *args, **kwargs):
        evaluator: Evaluator = Evaluator(None, self.syntax_tree,
                                         self.positive_traces_dir,
                                         self.negative_traces_dir,
                                         weight_size=0.33,
                                         weight_accuracy=0.66,
                                         initial_size=len(self.solution))

        if debug:
            print("Parsing true validation traces...")

        true_traces: List[PrefixTree] = PrefixTreeFactory. \
            parse_multiple_traces(self.positive_traces_dir, self.syntax_tree, True)

        if debug:
            print("Parsing false validation traces...")

        false_traces: List[PrefixTree] = PrefixTreeFactory. \
            parse_multiple_traces(self.negative_traces_dir, self.syntax_tree, True)

        evaluator.update(self.solution)
        e, _, _, _ = \
            evaluator.evaluate(true_trees=true_traces, false_trees=false_traces)

        histories: List[float] = []
        total_iterations: int = 0
        t: float = self.cooling_schedule.initial_temperature()

        for iteration in range(max_iterations):
            self.cooling_schedule.neighbourhood_size = len(self.solution.outgoing_edges)
            number_of_steps: int = self.cooling_schedule.chain_length()
            for m in range(number_of_steps):
                if len(self.solution) <= 3:
                    break
                if debug:
                    print(f"Iteration {iteration}, size: {len(self.solution)}, eval: {e}")
                neighbour: Graph = deepcopy(self.solution)
                self.selection.model = neighbour
                s1, s2 = self.selection.select()
                neighbour.merge_states(s1, s2, bool(random.getrandbits(1)))

                evaluator.update(neighbour)
                e1, _, _, _ = \
                    evaluator.evaluate(true_trees=true_traces, false_trees=false_traces)

                d: float = e1 - e

                if total_iterations < 100:
                    histories.append(abs(d))
                else:
                    histories[total_iterations % 100] = abs(d)

                acceptance_probability: float = exp((-d/mean(histories))/t)

                if d <= 0:
                    self.solution = neighbour
                    e = e1
                elif random.random() < acceptance_probability:
                    print(f"d={d}, histories={mean(histories)}, t={t}")
                    print(f"Acceptance prob: {acceptance_probability}")
                    self.solution = neighbour
                    e = e1

                total_iterations += 1
            t = self.cooling_schedule.update_temperature()
        return self.solution

    def initial_set(self, how_many: int) -> List[Graph]:
        res: List[Graph] = []

        for _ in range(how_many):
            y = deepcopy(self.solution)

            # Update the selection model and select two states to merge
            self.selection.update(y)
            s1, s2 = self.selection.select()

            # Perform the merge
            y.merge_states(s1, s2, bool(random.getrandbits(1)))

            res.append(y)

        return res

    @staticmethod
    def dominates(f: Tuple[float, float, float], s: Tuple[float, float, float]) -> bool:
        """
        Return true if f pareto-dominates s.
        """
        return f[0] <= s[0] and f[1] <= s[1] and f[2] <= s[2]

    @staticmethod
    def remove_dominated_solutions(solution_set: List[Graph],
                                   evaluations: List[Tuple[float, float, float]],
                                   new_solution: Graph,
                                   new_evaluation: Tuple[float, float, float],
                                   debug: bool = False) \
            -> Tuple[List[Graph], List[Tuple[float, float, float]]]:

        # Get list of dominated solutions
        dominated: List[int] = []
        for count, evaluation in enumerate(evaluations):
            if SimulatedAnnealing.dominates(new_evaluation, evaluation):
                dominated.append(count)

        # If any dominated solutions found
        if dominated:
            # Remove the dominated solutions from the solution set
            for index in reversed(dominated):
                if debug:
                    print(f"Solution dominated with eval {evaluations[index]}")
                solution_set.pop(index)
                evaluations.pop(index)

            # Add the better solution to the set
            if debug:
                print(f"Better solution with eval {new_evaluation}")
            evaluations.append(new_evaluation)
            solution_set.append(new_solution)
        else:
            is_dominated: bool = False
            for evaluation in evaluations:
                if SimulatedAnnealing.dominates(evaluation, new_evaluation):
                    is_dominated = True
                    break
            if not is_dominated:
                solution_set.append(new_solution)
                evaluations.append(new_evaluation)
                if debug:
                    print(f"{new_evaluation} does not dominate anything, but was added to the solution set.")

        return solution_set, evaluations

    @staticmethod
    def generate_random_weights() -> Tuple[float, float, float]:
        # Generate three numbers in [0,1]
        weights: Tuple[float, float, float] = (random.random(), random.random(), random.random())\

        # Normalize values
        s: float = weights[0] + weights[1] + weights[2]
        weights = (weights[0] / s, weights[1] / s, weights[2] / s)

        return weights

    @staticmethod
    def closest_non_dominated_solution(solution_set: List[Graph],
                                       evaluations: List[Tuple[float, float, float]],
                                       target: Graph,
                                       target_evaluation: Tuple[float, float, float]) \
            -> Union[None, Tuple[Graph, Tuple[float, float, float]]]:

        # Initial distance is infinity
        distance: float = float('inf')
        res_solution: Graph = None
        res_evaluation: Tuple[float, float, float] = None

        for solution, evaluation in zip(solution_set, evaluations):
            if solution is target:
                continue
            if not SimulatedAnnealing.dominates(target_evaluation, evaluation):
                if abs(len(target) - len(solution)) < distance:
                    res_solution = solution
                    res_evaluation = evaluation
        return res_solution, res_evaluation

    def search_mo(self,
                  front_size: int,
                  debug: bool = False,
                  max_interations: int = 11,
                  alpha: float = 1.05,
                  *args, **kwargs):
        M: List[Graph] = []
        S: List[Graph] = self.initial_set(front_size)

        if debug:
            print(f"Sizes: {len(self.solution)}, {list(map(lambda x: len(x), S))}")

        evaluations_front: List[Tuple[float, float, float]] = []
        evaluations_solutions: List[Tuple[float, float, float]] = []
        evaluator: Evaluator = Evaluator(None, self.syntax_tree,
                                         self.positive_traces_dir,
                                         self.negative_traces_dir,
                                         initial_size=len(self.solution))
        weights: Dict[Graph, Tuple[float, float, float]] = {}
        histories: Tuple[List[float], List[float], List[float]] = ([], [], [])

        # Initial temperature
        t: float = self.cooling_schedule.initial_temperature()

        # Outer iteration counter
        k: int = 0
        total_iterations: int = 0

        if debug:
            print("Parsing true validation traces...")

        true_traces: List[PrefixTree] = PrefixTreeFactory.\
            parse_multiple_traces(self.positive_traces_dir, self.syntax_tree, True)

        if debug:
            print("Parsing false validation traces...")

        false_traces: List[PrefixTree] = PrefixTreeFactory. \
            parse_multiple_traces(self.negative_traces_dir, self.syntax_tree, True)

        # For each initial solution
        for x in S:
            M.append(x)  # Add the solution to the potential efficient set
            evaluator.update(x)  # Evaluate the initial solution
            e2, sz, sp, re = evaluator.evaluate(true_trees=true_traces, false_trees=false_traces)
            evaluations_front.append((sz, sp, re))  # Add evaluation to the front
            evaluations_solutions.append((sz, sp, re))  # Add evaluation to the efficient set

        if debug:
            print(f"Starting annealing process with initial temperature {t}")

        while k < max_interations:
            vals: List[float] = []

            m = 0   # Inner iteration counter
            number_of_steps: int = self.cooling_schedule.chain_length()
            if debug:
                print(f"For iteration {k}, Solutions length: {len(self.solution)}")

            while m < number_of_steps:  # Enter Markov chain
                new_S: List[Graph] = []
                new_evaluations_front: List[Tuple[float, float, float]] = []

                for count, (x, evaluation) in enumerate(zip(S, evaluations_front)):
                    # Create a copy of the current solution
                    y: Graph = deepcopy(x)

                    # Update the selection model and select two states to merge
                    self.selection.update(y)

                    # Don't perform merges on the minimal state machine
                    if len(y) > 3:
                        s1, s2 = self.selection.select()

                        # Perform the merge
                        y.merge_states(s1, s2, bool(random.getrandbits(1)))

                    # Update the evaluator
                    evaluator.update(y)

                    # Evaluate the model and calculate the difference
                    e2, sz, sp, re = evaluator.evaluate(true_trees=true_traces, false_trees=false_traces)
                    new_evaluation: Tuple[float, float, float] = (sz, sp, re)

                    # Update histories:
                    if total_iterations < 100:
                        for c in range(3):
                            histories[c].append(abs(evaluation[c] - new_evaluation[c]))
                    else:
                        for c in range(3):
                            histories[c][total_iterations % 100] = abs(evaluation[c] - new_evaluation[c])

                    if debug:
                        print(f"Histories: {mean(histories)}")

                    if not self.dominates(evaluation, new_evaluation):
                        self.remove_dominated_solutions(M, evaluations_solutions,
                                                        y, new_evaluation, debug=debug)

                    if x not in weights:
                        weights[x] = self.generate_random_weights()
                        if debug:
                            print(f"New random weights because no x: {weights[x][0]}, {weights[x][1]}, {weights[x][2]}")
                    else:
                        # Select the closest non-dominated solution with respect to x
                        closest, closest_eval = self.closest_non_dominated_solution(S, evaluations_front, x, evaluation)
                        if closest is None:
                            weights[x] = self.generate_random_weights()
                            if debug:
                                print(f"New random weights because no x: {weights[x][0]}, {weights[x][1]}, {weights[x][2]}")
                        else:
                            s: float = 0
                            # Update the weights according to the rule
                            ws: List[float] = []
                            for c in range(3):
                                if evaluation[c] >= closest_eval[c]:
                                    ws.append(weights[x][c] * alpha)
                                else:
                                    ws.append(weights[x][c] / alpha)
                                s += ws[c]
                            # Normalize new weights
                            for c in range(3):
                                ws[c] /= s
                            weights[x] = (ws[0], ws[1], ws[2])
                            if debug:
                                print(f"Updated weights: {weights[x][0]}, {weights[x][1]}, {weights[x][2]}")

                    max_threshold: float = 1

                    for c in range(3):
                        threshold_c: float = exp(weights[x][c] *
                                                 (new_evaluation[c] - evaluation[c]) / t * (1/mean(histories[c])))
                        if debug:
                            print(f"Threshold {c}: {threshold_c}")
                        max_threshold = min(max_threshold, threshold_c)

                    acceptance_threshold: float = min(1.0, max_threshold)

                    if debug:
                        print(f"Iteration {k + 1} of {max_interations}, run {m} of {number_of_steps}\n "
                              f"solution {count} of {len(S)}: solution with eval {evaluation}\n "
                              f"neighbour with eval {new_evaluation}\n "
                              f"maximum threshold {max_threshold}\n "
                              f"members of M: {[id(x) for x in M]}")

                    # Accept a better solution with probability 1
                    # And a worse solution with probability e^(d/t)
                    if self.dominates(new_evaluation, evaluation) or random.random() < acceptance_threshold:
                        if debug:
                            print(f"Solution accepted, acceptance threshold {acceptance_threshold}\n"
                                  f"M size: {len(M)}")

                        # Update the next iteration front
                        new_S.append(y)
                        new_evaluations_front.append(new_evaluation)
                    else:
                        if debug:
                            print("Solution rejected.")

                        # Update the next iteration front
                        new_S.append(x)
                        new_evaluations_front.append(evaluation)

                S = new_S
                evaluations_front = new_evaluations_front
                # Increment the Markov chain counter
                m += 1
                total_iterations += 1

            # Increment the iteration counter
            k += 1

            # Update the temperature
            t = self.cooling_schedule.update_temperature(deviation=std(vals))
            self.cooling_schedule.neighborhood_size = max(map(lambda solution: len(solution), S))

            if debug:
                print(f"New temperature: {t}")
        return M


def kfcv_accuracy(syntax_tree: SyntaxTree,
                  logs_dir: str,
                  syntax_file_path: str,
                  output_file: str,
                  k: int = 5,
                  debug=False,
                  rs: bool=False):

    # helper function
    flatten = lambda l: [x for xs in l for x in xs]

    output_file = open(output_file, "a")
    if debug:
        print("Entering data split phase")

    # Check if log directory exists
    if not os.path.isdir(logs_dir):
        raise NotADirectoryError("Log directory not found!")

    model: Graph = PrefixTreeFactory.get_prefix_tree(traces_dir=logs_dir,
                                                     config_file_path=syntax_file_path,
                                                     remove_trivial_loops=False,
                                                     syntax_tree=syntax_tree)

    # Create directory names
    train_dir: str = logs_dir + "_train"

    validation_dir: str = logs_dir + "_validation"
    validation_dir_true: str = validation_dir + "_true"
    validation_dir_false: str = validation_dir + "_false"

    test_dir: str = logs_dir + "_test"
    test_dir_true: str = test_dir + "_true"
    test_dir_false: str = test_dir + "_false"

    os.mkdir(test_dir_true)
    os.mkdir(test_dir_false)
    os.mkdir(validation_dir_true)
    os.mkdir(validation_dir_false)
    os.mkdir(train_dir)

    # Get all traces and randomize their order
    traces: List[str] = os.listdir(logs_dir)
    random.shuffle(traces)

    # Split the trace names into equally sized chunks
    folds: List[List[str]] = array_split(traces, k)
    time_results = []
    for i in range(k):
        # Current test fold is i
        test_trace_num: int = i

        # Current validation fold is (i + 1) % k
        validation_trace_num: int = (i + 1) % k

        # All other folds are train traces
        train_traces_nums: List[int] = [x for x in range(k)
                                        if (x != test_trace_num and x != validation_trace_num)]

        test_traces = folds[test_trace_num]
        validation_traces = folds[validation_trace_num]
        train_traces = flatten([folds[x] for x in train_traces_nums])

        # For each train trace
        for train_trace in train_traces:

            # Create new name
            old_name = os.path.join(logs_dir, train_trace)
            new_name = os.path.join(train_dir, train_trace)

            # Move the trace
            os.rename(old_name, new_name)

        # For each validation trace
        for validation_trace_name in validation_traces:

            # Create new name
            old_name = os.path.join(logs_dir, validation_trace_name)
            new_name = os.path.join(validation_dir_true, validation_trace_name)

            # Create a false trace
            produce_false_trace(old_name,
                                os.path.join(validation_dir_false, validation_trace_name),
                                syntax_tree,
                                model)

            # Move the trace
            os.rename(old_name, new_name)

        # For each test trace
        for test_trace_name in test_traces:

            # Create new names
            old_name = os.path.join(logs_dir, test_trace_name)
            new_name = os.path.join(test_dir_true, test_trace_name)

            # Produce a false trace
            produce_false_trace(old_name,
                                os.path.join(test_dir_false, test_trace_name),
                                syntax_tree,
                                model)

            # Move the trace
            os.rename(old_name, new_name)

        # Minimize the model...
        initial_solution = PrefixTreeFactory.\
            get_prefix_tree(train_dir, cfg_file,
                            remove_trivial_loops=False,
                            one_state_per_template=True,
                            syntax_tree=syntax_tree)
        # UNCOMMENT FOR PSA!!!
        # schedule3 = LundySchedule(alpha=2.546e-1,
        #                           neighborhood_size=len(initial_solution.outgoing_edges),
        #                           sample_ratio=1 / 10)
        # sa = SimulatedAnnealing(initial_solution, s_tree,
        #                         validation_dir_true,
        #                         validation_dir_false,
        #                         TournamentSelection(initial_solution, 5, 0.75),
        #                         schedule3)

        # rs = RandomSearch(initial_solution, s_tree,
        #                   validation_dir_true,
        #                   validation_dir_false)

        # ls = TournamentLocalSearch(initial_solution, s_tree,
        #                            validation_dir_true,
        #                            validation_dir_false)

        prs = ParetoRandomSearch(initial_solution, s_tree,
                                 validation_dir_true, validation_dir_false)
        import time
        start = time.time()
        # UNCOMMENT FOR PSA!!!
        # solutions: List[Graph] = sa.search_mo(front_size=16,
        #                                       max_interations=10,
        #                                       debug=debug)

        # solutions: List[Graph] = [rs.search(debug=debug)]
        solutions: List[Graph] = prs.search_mo(front_size=16, debug=debug)

        end = time.time()
        time_results.append(end-start)

        evaluator: Evaluator = Evaluator(initial_solution, s_tree,
                                         test_dir_true, test_dir_false)

        true_trace_trees: List[PrefixTree] = PrefixTreeFactory. \
            parse_multiple_traces(test_dir_true, s_tree, True)
        false_trace_trees: List[PrefixTree] = PrefixTreeFactory. \
            parse_multiple_traces(test_dir_false, s_tree, True)

        output_file.write(f"Iteration {i} {len(solutions)} {end-start}\n")

        for solution in solutions:
            evaluator.update(solution)
            e2, sz, sp, re = evaluator.evaluate(true_trees=true_trace_trees,
                                                false_trees=false_trace_trees)
            output_file.write(f"{len(solution)} {e2} {sz} {sp} {re}\n")

        print("FINISHED ITERATION!!!!!!")

        # Clean up directories
        for train_trace in train_traces:
            old_name = os.path.join(logs_dir, train_trace)
            new_name = os.path.join(train_dir, train_trace)
            os.rename(new_name, old_name)
        for validation_trace_name in validation_traces:
            old_name = os.path.join(logs_dir, validation_trace_name)
            new_name = os.path.join(validation_dir_true, validation_trace_name)
            false_trace = os.path.join(validation_dir_false, validation_trace_name)
            os.remove(false_trace)
            os.rename(new_name, old_name)
        for test_trace_name in test_traces:
            old_name = os.path.join(logs_dir, test_trace_name)
            new_name = os.path.join(test_dir_true, test_trace_name)
            false_trace = os.path.join(test_dir_false, test_trace_name)
            os.remove(false_trace)
            os.rename(new_name, old_name)

    os.rmdir(test_dir_true)
    os.rmdir(test_dir_false)
    os.rmdir(validation_dir_true)
    os.rmdir(validation_dir_false)
    os.rmdir(train_dir)
    output_file.close()
    print(np.mean(time_results))


def evaluate_accuracy(syntax_tree: SyntaxTree,
                      logs_dir: str,
                      syntax_file_path: str,
                      k: int = 5,
                      debug=False,
                      rs: bool=False):

    if not os.path.isdir(logs_dir):
        raise NotADirectoryError("Log directory not found!")

    # Get all traces in the directory
    traces: List[str] = os.listdir(logs_dir)

    number_of_traces: int = 100

    while number_of_traces <= 1000:

        print(f"Starting accuracy test with {number_of_traces} traces.")

        seed: int = number_of_traces
        random.seed(seed + 1)

        # Select the next pool of traces
        current_trace_pool = random.sample(traces, k=number_of_traces)

        # The directory for the selected traces
        new_dir: str = logs_dir + f"_kfcv_{number_of_traces}"
        os.mkdir(new_dir)

        for current_trace in current_trace_pool:
            # Create new name
            old_name = os.path.join(logs_dir, current_trace)
            new_name = os.path.join(new_dir, current_trace)

            # Move the trace
            os.rename(old_name, new_name)

        start = time.time()
        file_to_write: str = f"kfcv_{number_of_traces}_output.txt"
        if rs:
            file_to_write = "prs_" + file_to_write
        # Perform KFCV here
        kfcv_accuracy(
            syntax_tree,
            new_dir,
            syntax_file_path,
            file_to_write,
            k=k,
            debug=debug,
            rs=rs)

        end = time.time()

        print(f"Iteration took {end-start} seconds.")

        # Move the traces back into the original directory
        for current_trace in current_trace_pool:
            # Create new name
            old_name = os.path.join(logs_dir, current_trace)
            new_name = os.path.join(new_dir, current_trace)

            # Move the trace
            os.rename(new_name, old_name)

        print(f"Finished KFCV with {number_of_traces} traces.")

        # Remove the empty directory
        os.rmdir(new_dir)

        number_of_traces += 50


def evaluate_scalability(syntax_tree: SyntaxTree,
                         logs_dir: str,
                         syntax_file_path: str,
                         output_file: str,
                         debug=False):

    # helper function
    flatten = lambda l: [x for xs in l for x in xs]

    # Partition size
    k = 5

    output_file = open(output_file, "a")
    if debug:
        print("Entering data split phase")

    # Check if log directory exists
    if not os.path.isdir(logs_dir):
        raise NotADirectoryError("Log directory not found!")

    model: Graph = PrefixTreeFactory.get_prefix_tree(traces_dir=logs_dir,
                                                     config_file_path=syntax_file_path,
                                                     remove_trivial_loops=False,
                                                     syntax_tree=syntax_tree)

    # Create directory names
    train_dir: str = logs_dir + "_train"

    validation_dir: str = logs_dir + "_validation"
    validation_dir_true: str = validation_dir + "_true"
    validation_dir_false: str = validation_dir + "_false"

    os.mkdir(validation_dir_true)
    os.mkdir(validation_dir_false)
    os.mkdir(train_dir)

    # Get all traces and randomize their order
    traces: List[str] = os.listdir(logs_dir)
    random.shuffle(traces)

    # Split the trace names into equally sized chunks
    folds: List[List[str]] = array_split(traces, k)

    # Assign first partition to the validation set
    validation_trace_num: int = 0

    # Assign all other partitions to the train set
    train_traces_nums: List[int] = list(range(1, k))

    validation_traces = folds[validation_trace_num]
    train_traces = flatten([folds[x] for x in train_traces_nums])

    # For each train trace
    for train_trace in train_traces:

        # Create new name
        old_name = os.path.join(logs_dir, train_trace)
        new_name = os.path.join(train_dir, train_trace)

        # Move the trace
        os.rename(old_name, new_name)

    # For each validation trace
    for validation_trace_name in validation_traces:

        # Create new name
        old_name = os.path.join(logs_dir, validation_trace_name)
        new_name = os.path.join(validation_dir_true, validation_trace_name)

        # Create a false trace
        produce_false_trace(old_name,
                            os.path.join(validation_dir_false, validation_trace_name),
                            syntax_tree,
                            model)

        # Move the trace
        os.rename(old_name, new_name)

    # Minimize the model...
    initial_solution = PrefixTreeFactory.\
        get_prefix_tree(train_dir, cfg_file,
                        remove_trivial_loops=False,
                        one_state_per_template=True,
                        syntax_tree=syntax_tree)

    schedule3 = LundySchedule(alpha=2.546e-1,
                              neighborhood_size=len(initial_solution.outgoing_edges),
                              sample_ratio=1 / 10)

    sa = SimulatedAnnealing(initial_solution, s_tree,
                            validation_dir_true,
                            validation_dir_false,
                            TournamentSelection(initial_solution, 5, 0.75),
                            schedule3)

    # rs = RandomSearch(initial_solution, s_tree,
    #                   validation_dir_true,
    #                   validation_dir_false)

    # ls = TournamentLocalSearch(initial_solution, s_tree,
    #                            validation_dir_true,
    #                            validation_dir_false)

    start = time.time()

    # solutions: List[Graph] = sa.search_mo(front_size=16,
    #                                       max_interations=10,
    #                                       debug=debug)

    solutions: List[Graph] = [sa.search()]
    end = time.time()

    output_file.write(f"Iteration {len(solutions)} {end-start}\n")

    for solution in solutions:
        output_file.write(f"{len(solution)}\n")

    # Clean up directories
    for train_trace in train_traces:
        old_name = os.path.join(logs_dir, train_trace)
        new_name = os.path.join(train_dir, train_trace)
        os.rename(new_name, old_name)
    for validation_trace_name in validation_traces:
        old_name = os.path.join(logs_dir, validation_trace_name)
        new_name = os.path.join(validation_dir_true, validation_trace_name)
        false_trace = os.path.join(validation_dir_false, validation_trace_name)
        os.remove(false_trace)
        os.rename(new_name, old_name)

    os.rmdir(validation_dir_true)
    os.rmdir(validation_dir_false)
    os.rmdir(train_dir)
    output_file.close()


def evaluate_scalability_all(syntax_tree: SyntaxTree,
                             logs_dir: str,
                             syntax_file_path: str,
                             k: int = 5,
                             debug=False):
    if not os.path.isdir(logs_dir):
        raise NotADirectoryError("Log directory not found!")

    # Get all traces in the directory
    traces: List[str] = os.listdir(logs_dir)

    number_of_traces: int = 950

    while number_of_traces <= 1000:

        print(f"Starting KFCV with {number_of_traces} traces.")

        seed: int = number_of_traces + 1
        random.seed(seed)

        # The directory for the selected traces
        new_dir: str = logs_dir + f"_scalability_{number_of_traces}"
        os.mkdir(new_dir)

        i_start = time.time()

        # Perform k tests
        for i in range(k):

            start = time.time()

            # Select the next pool of traces
            current_trace_pool = random.sample(traces, k=number_of_traces)

            for current_trace in current_trace_pool:
                # Create new name
                old_name = os.path.join(logs_dir, current_trace)
                new_name = os.path.join(new_dir, current_trace)

                # Move the trace
                os.rename(old_name, new_name)

            # Perform scalability test here
            evaluate_scalability(
                syntax_tree,
                new_dir,
                syntax_file_path,
                f"sosa_scalability_{number_of_traces}_output.txt",
                debug=debug)

            # Move the traces back into the original directory
            for current_trace in current_trace_pool:
                # Create new name
                old_name = os.path.join(logs_dir, current_trace)
                new_name = os.path.join(new_dir, current_trace)

                # Move the trace
                os.rename(new_name, old_name)

            end = time.time()
            print(f"Iteration {i + 1} out of {k} took {end-start} seconds.")

        os.rmdir(new_dir)

        i_end = time.time()

        print(f"Finished scalability test for {number_of_traces} traces. It took {i_end - i_start}")

        number_of_traces += 50


if __name__ == '__main__':

    project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
    cfg_file = str(project_root.joinpath('resources/config.json'))
    all_logs = str(project_root.joinpath('resources/logs'))
    s_tree = SyntaxTreeFactory().parse_file(cfg_file)
    output_file = "output_kfold_test2.txt"
    evaluate_accuracy(s_tree,
                      all_logs,
                      cfg_file,
                      k=5,
                      debug=True,
                      rs=True)
    # evaluate_scalability_all(s_tree,
    #                          all_logs,
    #                          cfg_file,
    #                          k=5,
    #                          debug=False)
