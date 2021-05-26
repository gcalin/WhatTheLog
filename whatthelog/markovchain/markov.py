import multiprocessing
import os
import random
import sys
from copy import deepcopy
import shutil
from functools import partial
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
        self.maxLength = 0
        self.tracesdir = tracesdir
        self.config_file = config_file
        self.falsedir = 'out/false_traces/'
        self.truedir = 'resources/truetraces/'
        self.truetestdir = 'out/truetraces/'

        self.weight_size = weight_size
        self.weight_accuracy = weight_accuracy

        # Parse the syntaxtree from the config file.
        self.syntaxtree: SyntaxTree = SyntaxTreeFactory().parse_file(self.config_file)
        all_states: List[str] = self.get_all_states(self.syntaxtree)
        self.pt = None

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

        self.initial_size = len(self.transitionMatrix)

        print('[ markov.py ] - Chain ready.')

    def generate_false_traces_from_prefixtree(self, amount: int = 500):
        self.pt = PrefixTreeFactory.get_prefix_tree(self.tracesdir, self.config_file)
        print('[ markov.py ] - Generating false traces...')
        self.generate_false_traces(amount)
        print('[ markov.py ] - False traces generated')

    def generate_false_traces(self, amount):
        pbar = tqdm(range(amount), file=sys.stdout, leave=False)
        for i in pbar:
            produce_false_trace(os.path.join(self.tracesdir, os.listdir(self.tracesdir)
            [random.randint(0, len(os.listdir(self.tracesdir)) - 1)]),
                                self.falsedir + '/false' + str(i), self.syntaxtree, self.pt)

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

        self.initial_size = len(self.transitionMatrix)

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
                if self.transitionMatrix[r][d] >= 1.0 - threshold and r != d:
                    temporary_result.append([r, d])  # Only one, otherwise things might break
                    break

        result = list()

        for i in temporary_result:
            new = True
            for r in result:
                if r[-1] == i[0] and i[1] not in r:
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
        if self.weight_size * len(candidate) / self.maxLength + self.weight_accuracy < self.weight_size:
            # Optimal outcome for this candidate vs the worst outcome for the candidate with most compression
            print('=========================================================')
            return 0
        else:
            matrix = deepcopy(self.transitionMatrix)
            states = deepcopy(self.states)
            current_length = len(candidate)
            self.process_candidate_list(candidate, matrix, states)
            return self.weight_size * current_length / self.maxLength + self.weight_accuracy * \
                   sum(self.calculate_accuracy(matrix, states)) / 3

    def calculate_accuracy(self, matrix=None, states=None):  # specificity
        if matrix is None:
            matrix = self.transitionMatrix
        if states is None:
            states = self.states

        true_negative = 0
        for file in os.listdir(self.falsedir):
            with open(os.path.join(self.falsedir, file), 'r') as f:
                current = states['root']
                for l in f.readlines():
                    next_node = states[self.syntaxtree.search(l).name]
                    if matrix[current][next_node] < 1 / 1e10:  # Some small value close to 0
                        true_negative += 1
                        break
                    current = next_node

        true_positive = 0
        for file in os.listdir(self.truetestdir):
            with open(os.path.join(self.truetestdir, file), 'r') as f:
                is_valid = True
                current = states['root']
                for l in f.readlines():
                    if self.syntaxtree.search(l).name not in states:
                        is_valid = False
                        break
                    next_node = states[self.syntaxtree.search(l).name]
                    if matrix[current][next_node] < 1 / 1e10:  # Some small value close to 0
                        is_valid = False
                        break
                    current = next_node
                if is_valid:
                    true_positive += 1

        specificity = true_negative / len(os.listdir(self.falsedir))
        recall = true_positive / len(os.listdir(self.truetestdir))
        precision = true_positive / (true_positive + (len(os.listdir(self.falsedir)) - true_negative))

        return specificity, recall, precision

    def get_candidates(self, threshold: float = 0.0):
        # Duplicate rows
        candidates = list(map(lambda x: sorted(x), self.find_duplicates(threshold)))
        # Duplicate columns
        candidates += list(map(lambda x: sorted(x), list(
            filter(lambda x: sorted(x) not in candidates, self.find_duplicates(threshold, False)))))
        # Probability of 1
        candidates += list(
            map(lambda x: sorted(x), list(filter(lambda x: sorted(x) not in candidates, self.find_prop_1(threshold)))))

        return candidates

    def do_it(self, size: int):
        assert size > 0, 'Can not have a size of 0'

        self.train()

        while len(self.transitionMatrix) > size:
            threshold = 0.0
            print(str(100 * (self.initial_size - len(self.transitionMatrix)) / (self.initial_size - size)) + ' %')
            candidates = []
            while len(candidates) == 0:  # move boundaries to find more candidates
                candidates = self.get_candidates(threshold)
                threshold += 0.001

            to_split = []
            for c in range(len(candidates)):
                if len(candidates[c]) - 1 > len(self.transitionMatrix) - size:
                    to_split.append(c)

            to_split.reverse()
            for s in to_split:
                l = candidates[s]
                i = 1
                while i < len(l):
                    candidates.append([l[i-1], l[i]])
                    i += 1
                del candidates[s]

            self.maxLength = 0
            for c in candidates:
                c.sort()
                if len(c) > self.maxLength:
                    self.maxLength = len(c)

            max = 0
            if len(candidates) > 1:
                a_pool = multiprocessing.Pool(processes=12)
                results = a_pool.map(self.evaluate_candidate, candidates)
                a_pool.close()
                a_pool.join()

                for r in range(len(candidates)):
                    if results[r] > results[max]:
                        max = r

            # self.print_matrix()
            self.process_candidate_list(candidates[max])
            with open('out/eval/evaluation4', 'a') as f:
                specificity, recall, precision = self.calculate_accuracy()
                f.write('<tr><td>' + str(1 - len(self.transitionMatrix)/self.initial_size) + '</td>')
                f.write('<td>' + str(specificity) + '</td>')
                f.write('<td>' + str(recall) + '</td>')
                f.write('<td>' + str(precision) + '</td></tr>\n')
        # self.print_matrix()

    def print_matrix(self, matrix=None):
        if matrix is None:
            matrix = self.transitionMatrix
        for r in range(len(matrix)):
            vstr = str(r) + ": "
            for c in range(len(matrix)):
                vstr += str(matrix[r][c]) + " "
            print(vstr)

        [print(i, aaa) for aaa, i in self.states.items()]

    def select_true_traces(self, amount=50):
        if os.path.exists(self.truetestdir):
            shutil.rmtree(self.truetestdir)
        os.mkdir(self.truetestdir)
        for i in range(amount):
            shutil.copy(os.path.join(self.truedir, os.listdir(self.truedir)
            [random.randint(0, len(os.listdir(self.truedir)) - 1)]),
                        os.path.join(self.truetestdir, os.listdir(self.truedir)
            [random.randint(0, len(os.listdir(self.truedir)) - 1)]))

if __name__ == '__main__':
    # self.train("../../tests/resources/testlogs/")
    # self.train("../../resources/traces/")
    # self.train("../../../all/")
    # chain = MarkovChain("resources/traces/", weight_size=0.8, weight_accuracy=0.2)
    # direc = "resources/traces5/"
    # direc = "tests/resources/testlogs/"
    # chain = MarkovChain(direc, weight_size=0.5, weight_accuracy=0.5)

    # falsetraces = MarkovChain(direc)
    # falsetraces.train()
    # print(len(falsetraces.transitionMatrix))
    # falsetraces.generate_false_traces_from_prefixtree(50)
    # chain.do_it(6)
    # chain.print_matrix()
    # print(chain.calculate_accuracy())

    with open('out/eval/evaluation4', 'w+') as f:
        f.write('')
    for j in range(10):
        # if j > 8:
            print('current progress:', j+1, '/ 10')
            # falsetraces.generate_false_traces(50);

            direc = 'resources/traces' + str(j + 1) + '/'
            falsetraces = MarkovChain(direc)
            falsetraces.select_true_traces(100)
            falsetraces.generate_false_traces_from_prefixtree(100)

            chain = MarkovChain(direc, weight_size=0.5, weight_accuracy=0.5)
            chain.train()
            # print(len(chain.transitionMatrix))
            chain.do_it(1)


    # chain.do_it(6)
    # print(chain.calculate_score())
