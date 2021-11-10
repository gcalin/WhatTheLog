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

                acceptance_probability: float = exp((-d / mean(histories)) / t)

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
        return f[0] <= s[0] and f[1] <= s[1] and f[2] <= s[2] \
               and (f[0] < s[0] or f[1] < s[1] or f[2] < s[2])

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
        weights: Tuple[float, float, float] = (random.random(), random.random(), random.random())

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

        true_traces: List[PrefixTree] = PrefixTreeFactory. \
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

            m = 0  # Inner iteration counter
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
                                print(
                                    f"New random weights because no x: {weights[x][0]}, {weights[x][1]}, {weights[x][2]}")
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
                                                 (new_evaluation[c] - evaluation[c]) / t * (1 / mean(histories[c])))
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
