import multiprocessing
import os
import random
import sys
from copy import deepcopy
from typing import List, Dict

import numpy as np
from tqdm import tqdm

from scripts.log_scrambler import produce_false_trace
from whatthelog.prefixtree.edge_properties import EdgeProperties
from whatthelog.prefixtree.evaluator import Evaluator
from whatthelog.prefixtree.graph import Graph
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.prefixtree.state import State
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory

random.seed(os.environ['random_seed'] if 'random_seed' in os.environ else 5)


class MarkovChain:

    def __init__(self, tracesdir: str, config_file: str = "resources/config.json", weight_size: float = 0.5,
                 weight_accuracy: float = 0.5):
        self.tracesdir = tracesdir
        self.config_file = config_file
        self.falsedir = 'out/false_traces/'

        self.weight_size = weight_size
        self.weight_accuracy = weight_accuracy
        self.initial_size = -1

        # Parse the syntaxtree from the config file.
        self.syntaxtree: SyntaxTree = SyntaxTreeFactory().parse_file(self.config_file)
        all_states: List[str] = self.get_all_states(self.syntaxtree)

        self.graph = None  # This is a storage for the current state for the parallel execution
        self.graph_nodes = None
        # self.evaluator = Evaluator(None,
        #                            self.syntaxtree,
        #                            self.tracesdir,
        #                            self.falsedir,
        #                            initial_size=len(all_states),
        #                            weight_size=weight_size,
        #                            weight_accuracy=weight_accuracy)

        # Give every state an id.
        self.states: Dict[str, int] = dict()
        self.states['root'] = 0
        self.states['terminal'] = 1
        for a in range(len(all_states)):
            if all_states[a] in self.states:
                raise Exception(all_states[a])
            self.states[all_states[a]] = a + 2

        # Initialize the transition matrix for the root and terminal node.
        self.transitionMatrix = [[0 for _ in range(len(self.states))]
                                 for _ in range(len(self.states))]

        # todo remove unreached nodes

        print('[ markov.py ] - Chain ready.')

    def generate_false_traces(self, amount: int = 500):
        syntax_tree = SyntaxTreeFactory().parse_file(self.config_file)
        pt = PrefixTreeFactory.get_prefix_tree(self.tracesdir, self.config_file)
        print('[ markov.py ] - Generating false traces...')
        pbar = tqdm(range(amount), file=sys.stdout, leave=False)
        for i in pbar:
            produce_false_trace(os.path.join(self.tracesdir, os.listdir(self.tracesdir)
            [random.randint(0, len(os.listdir(self.tracesdir)) - 1)]),
                                self.falsedir + '/false' + str(i), syntax_tree, pt)
        print('[ markov.py ] - False traces generated')

    def get_all_states(self, syntaxtree: SyntaxTree) -> List[str]:
        res = []
        if len(syntaxtree.get_children()) > 0:
            for c in syntaxtree.get_children():
                res += self.get_all_states(c)
        else:
            return [syntaxtree.name]
        return res

    def remove(self, new: int, to_delete: int, matrix=None, states=None) -> None:
        if new == to_delete:
            raise Exception('trying to delete self')

        if matrix is None:
            matrix = self.transitionMatrix
        if states is None:
            states = self.states

        for i in range(len(matrix)):
            matrix[new][i] += matrix[to_delete][i]
            matrix[new][i] /= 2  # Normalize again

        del matrix[to_delete]

        for key in states.copy():
            if states[key] == to_delete:
                states[key] = new
            elif states[key] > to_delete:
                states[key] -= 1

        for arr in matrix:
            arr[new] += arr[to_delete]
            del arr[to_delete]

    def parallel(self, files: List[str]):
        for file in files:
            with open(self.tracesdir + file, 'r') as f:
                current = 'root'
                for line in f.readlines():
                    try:
                        next_node = self.syntaxtree.search(line).name
                    except AttributeError as e:
                        print(line)
                        raise e
                    if next_node not in self.states:
                        raise Exception('Node not found in self.states')
                    self.transitionMatrix[self.states[current]][self.states[next_node]] += 1
                    current = next_node
                self.transitionMatrix[self.states[current]][self.states['terminal']] += 1
        return self.transitionMatrix

    def delete_unreachable(self):
        to_delete: List[int] = list()
        for c in range(len(self.transitionMatrix)):
            if c > 0:  # the root state is an exception
                s = 0
                for r in self.transitionMatrix:
                    s += r[c]
                if s == 0:
                    to_delete.append(c)

        to_delete.sort(reverse=True)
        for d in to_delete:
            for key in self.states.copy():
                if self.states[key] == d:
                    del self.states[key]
                elif self.states[key] > d:
                    self.states[key] -= 1
            del self.transitionMatrix[d]
            for r in self.transitionMatrix:
                del r[d]

    def train(self):
        print('[ markov.py ] - Start training using multiprocessing...')

        a_pool = multiprocessing.Pool()
        matrix = None
        for result in a_pool.map(self.parallel, np.array_split(os.listdir(self.tracesdir), 12)):
            if matrix is None:
                matrix = result
            else:
                for r in range(len(result)):
                    for c in range(len(result)):
                        matrix[r][c] += result[r][c]
        a_pool.close()
        a_pool.join()
        self.transitionMatrix = matrix
        print('[ markov.py ] - Training done.')

        print('[ markov.py ] - Start Normalizing...')
        for i in range(len(self.transitionMatrix)):
            row = self.transitionMatrix[i]
            s = sum(row)
            if s > 0:
                for a in range(len(row)):
                    self.transitionMatrix[i][a] /= s
        print('[ markov.py ] - Normalizing done.')

        print('[ markov.py ] - Removing unreachable states...')
        self.delete_unreachable()
        print('[ markov.py ] - Unreachable states removed.')

        self.transitionMatrix[1][1] = 1  # Create a self loop for the terminal state

    def find_duplicates(self, threshold: float = 0.0, rowdup: bool = True) -> List[List[int]]:
        result = list()

        for row in range(len(self.transitionMatrix)):
            to_append = [row]
            for i in range(len(self.transitionMatrix)):
                if row != i:

                    found = False
                    for a in result:
                        if i in a:
                            found = True
                            break
                    if not found:
                        equivalent = True
                        for pos in range(len(self.transitionMatrix[i])):
                            if (rowdup and
                                abs(self.transitionMatrix[i][pos] - self.transitionMatrix[row][pos]) > threshold) \
                                    or \
                                    (not rowdup and
                                     abs(self.transitionMatrix[pos][i] - self.transitionMatrix[pos][row]) > threshold):
                                equivalent = False
                                break
                        if equivalent:
                            to_append.append(i)
            if len(to_append) > 1:
                result.append(to_append)

        return result

    def find_prop_1(self, threshold: float = 0.0) -> List[List[int]]:
        temporary_result = list()

        for r in range(len(self.transitionMatrix)):
            for d in range(len(self.transitionMatrix)):
                # We don't want to remove the root and terminal node
                if self.transitionMatrix[r][d] >= 1.0 - threshold and r not in [0, 1] and d not in [0, 1] and r != d:
                    temporary_result.append([r, d])  # Only one, otherwise things might break
                    break

        result = list()

        for i in temporary_result:
            new = True
            for r in result:
                if r[-1] == i[0]:
                    r.append(i[1])
                    new = False
                    break
            if new:
                result.append(i)

        return result

    def build_graph(self) -> Graph:
        self.graph_nodes = dict()
        self.graph_nodes[0] = State(['root'])
        graph = Graph(self.graph_nodes[0])
        graph.add_state(self.graph_nodes[0])
        for k, v in self.states.items():
            if k != 'root':
                self.graph_nodes[v] = State([k])
                graph.add_state(self.graph_nodes[v])

        for r in range(len(self.transitionMatrix)):
            for c in range(len(self.transitionMatrix)):
                if self.transitionMatrix[r][c] > 0:
                    graph.add_edge(self.graph_nodes[r], self.graph_nodes[c], EdgeProperties([]))
        return graph

    def process_candidate_list(self, candidate: List[int], matrix=None, states=None):
        current = candidate.pop(0)
        while len(candidate) > 0:
            next_node = candidate.pop(0)
            self.remove(current, next_node, matrix, states)
            for a in range(len(candidate)):
                candidate[a] -= 1

    def evaluate_candidate(self, candidate: List[int]):
        matrix = deepcopy(self.transitionMatrix)
        states = deepcopy(self.states)
        self.process_candidate_list(candidate, matrix, states)

        return self.calculate_score(matrix, states)

    def calculate_score(self, matrix=None, states=None):  # specificity
        if matrix is None:
            matrix = self.transitionMatrix
        if states is None:
            states = self.states
        true_negative = 0
        false_positive = 0
        for file in os.listdir(self.falsedir):
            with open(os.path.join(self.falsedir, file), 'r') as f:
                current = states['root']
                detect_false = False
                for l in f.readlines():
                    next_node = states[self.syntaxtree.search(l).name]
                    if matrix[current][next_node] < 1 / 1e6:  # Some small value
                        true_negative += 1
                        detect_false = True
                        break
                    current = next_node
                if not detect_false:
                    false_positive += 1

        specificity = true_negative / (true_negative + false_positive)

        # recall
        true_positive = 0
        false_negative = 0
        for file in os.listdir(self.tracesdir):  # todo use others
            with open(os.path.join(self.tracesdir, file), 'r') as f:
                current = states['root']
                detect_false = False
                for l in f.readlines():
                    next_node = states[self.syntaxtree.search(l).name]
                    if matrix[current][next_node] < 1 / 1e6:  # Some small value
                        false_negative += 1
                        detect_false = True
                        break
                    current = next_node
                if not detect_false:
                    true_positive += 1

        recall = true_positive / (false_negative + true_positive)

        accuracy = (specificity + recall) / 2

        size = 1 - len(matrix) / self.initial_size
        print(size, specificity, recall)
        return self.weight_size * size + self.weight_accuracy * accuracy

    def do_it(self, size: int):
        assert size > 0, 'Can not have a size of 0'

        self.train()

        threshold = 0.0
        if self.initial_size == -1:
            self.initial_size = len(self.transitionMatrix)
        while len(self.transitionMatrix) > size:
            print(str(100 * (self.initial_size - len(self.transitionMatrix)) / (self.initial_size - size)) + ' %')
            candidates = self.find_duplicates(threshold) + self.find_prop_1(threshold) + self.find_duplicates(threshold, False)
            while len(candidates) == 0:  # move boundaries to find more candidates
                threshold += 0.001
                candidates = self.find_duplicates(threshold) + self.find_prop_1(threshold) + self.find_duplicates(threshold, False)
            for c in candidates:
                c.sort()
            # print(candidates)
            a_pool = multiprocessing.Pool()
            results = a_pool.map(self.evaluate_candidate, candidates)
            a_pool.close()
            a_pool.join()
            # print(results)
            max = 0
            for r in range(len(results)):
                if results[r] > results[max]:
                    max = r
            # print(candidates[max])
            # print(self.states)
            # self.print_matrix()
            # print('---------')
            self.process_candidate_list(candidates[max])
        print('')
        self.print_matrix()

        # todo should create the false traces from unused log files

    def print_matrix(self, matrix=None):
        if matrix is None:
            matrix = self.transitionMatrix
        for r in range(len(matrix)):
            vstr = str(r) + ": "
            for c in range(len(matrix)):
                # if self.transitionMatrix[r][c] == 0.0:
                #     vstr += "    "
                # else:
                vstr += str(matrix[r][c]) + " "
            print(vstr)

        [print(i, aaa) for aaa, i in self.states.items()]


if __name__ == '__main__':
    # self.train("../../tests/resources/testlogs/")
    # self.train("../../resources/traces/")
    # self.train("../../../all/")
    # chain = MarkovChain("resources/traces/", weight_size=0.8, weight_accuracy=0.2)
    chain = MarkovChain("tests/resources/testlogs/", weight_size=0.8, weight_accuracy=0.2)

    # chain.generate_false_traces(500)

    # todo hard because
    #  lot of duplicate checking
    #  A LOT OF evaluating

    chain.do_it(5)
    print(chain.calculate_score())
