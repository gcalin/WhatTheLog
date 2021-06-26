import multiprocessing
import os
from enum import Enum
from typing import List, Tuple
from tqdm import tqdm

from scipy.sparse import lil_matrix
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory

from whatthelog.prefixtree.visualizer import Visualizer
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory

import sys
class MarkovChain:
    SINGULAR_LOOPS = 1
    ALL_POSSIBLE_LOOPS = 2
    ONLY_PROBABILISTIC = 3

    def __init__(self, tree_type: int):
        self.tree_type = tree_type
        self.terminal_index = -1
        config_file = "../../resources/config.json"
        self.syntaxtree = SyntaxTreeFactory().parse_file(config_file)
        # self.prefixtree = PrefixTreeFactory.get_prefix_tree("../../tests/resources/testlogs/", config_file)
        self.prefixtree = PrefixTreeFactory.get_prefix_tree("../../resources/traces/", config_file)
        self.prefixtree.add_terminal()
        # self.states = dict()
        # self.current = 0
        # self.current_log = 0
        # self.log_templates = dict()
        # self.count_states()
        # Visualizer(self.prefixtree).visualize('test.png')
        self.prefixtree.remove_loops(self.tree_type == MarkovChain.ALL_POSSIBLE_LOOPS)

        self.states = dict()
        self.current = 0
        self.current_log = 0
        self.log_templates = dict()
        self.count_states()
        # self.transitionMatrix = [[0 for _ in range(count)]
        #                          for _ in range(count)]
        if tree_type == MarkovChain.ONLY_PROBABILISTIC:
            self.transitionMatrix: lil_matrix = lil_matrix((len(self.log_templates), len(self.log_templates)), dtype=float)
        else:
            self.transitionMatrix: lil_matrix = lil_matrix((self.current, self.current), dtype=float)

        print('[ markovchain_old.py ] - Prefix Graph ready.')

    # def remove(self, duplicate: int, new: int) -> None:
    #     del self.transitionMatrix[duplicate]
    #     for arr in self.transitionMatrix:
    #         arr[new] = arr[new] + arr[duplicate]
    #         del arr[duplicate]
    #
    # def merge_subsequent(self):
    #     for r in range(len(self.transitionMatrix[0]) - 1):
    #         # print('R: ', r)
    #         # print(len(self.transitionMatrix[r]))
    #         if max(self.transitionMatrix[r]) == 1.0:  # Always goes to the same next node
    #             for i in range(len(self.transitionMatrix[r]) - 1):
    #                 if self.transitionMatrix[i] == 1.0:
    #                     break
    #             self.remove(r, i)
    #             return

    def count_states(self):

        print('[ markovchain_old.py ] - Counting states...')
        current = self.prefixtree.start_node
        been = set()
        stack = [current]
        while len(stack) > 0:
            current = stack.pop()
            if current.is_terminal:
                self.terminal_index = self.current
            current.id = self.current
            self.current += 1

            if current.properties.log_templates[0] not in self.log_templates.keys():
                self.log_templates[current.properties.log_templates[0]] = self.current_log
                self.current_log += 1

            outgoing = self.prefixtree.get_outgoing_states(current)
            been.add(current)
            for node in [n for n in outgoing if n not in been]:
                stack.append(node)









        # children = [x for x in self.prefixtree.get_children(state) if x not in been]
        # if children:
        #     self.current += 1
            # the_end = list(filter(lambda x: x.is_terminal, children))
            # for child in children:
            #     self.count_states_rec(child, been)

    def train(self, directory: str):
        print('[ markovchain_old.py ] - Start training...')

        pbar = tqdm(os.listdir(directory), file=sys.stdout, leave=False)
        for filename in pbar:
            with open(directory + filename, 'r') as f:
                queue = [(self.prefixtree.get_root(), [self.prefixtree.get_root().id])]
                for line in f.readlines():

                    if self.tree_type == MarkovChain.ONLY_PROBABILISTIC:
                        next_template = self.syntaxtree.search(line).name
                        _, path = queue.pop(0)
                        path.append(self.log_templates[next_template])
                        queue.append((None, path))
                    else:
                        newqueue = list()
                        while len(queue) > 0:
                            current, path = queue.pop(0)

                            next_template = self.syntaxtree.search(line).name
                            for child in self.prefixtree.get_children(current):
                                if child.properties.log_templates[0] == next_template:
                                    newpath = path.copy()
                                    newpath.append(child.id)
                                    # todo memory issue here?
                                    # It grows non-deterministically
                                    newqueue.append((child, newpath))
                        queue = newqueue
                    if len(queue) > 1:
                        print(queue[0])
                        print(queue[1])
                        raise Exception('Non-determinism detected', str(len(queue)))

                print('GOT ALL PATHS:', f.name)
                for _, path in queue:
                    value = 1 / len(queue)
                    for i in range(len(path) - 1):
                        self.transitionMatrix[path[i], path[i+1]] += value
                    print(path[i+1], self.terminal_index)
                    print(self.transitionMatrix)
                    self.transitionMatrix[path[i+1], self.log_templates['markov-terminal']] += value
                # print(queue)
                # print(len(queue))
                # return

                    #
                    # if self.only_unique:
                    #     self.transitionMatrix[self.log_templates[parent.properties.log_templates[0]],
                    #                           self.log_templates[child.properties.log_templates[0]]] += 1
                    # else:
                    #     self.transitionMatrix[parent.id, child.id] += 1
                    # parent = child

        print('[ markovchain_old.py ] - Start Normalizing...')
        for r in range(self.transitionMatrix.shape[0]):
            row = self.transitionMatrix.getrow(r)
            s = sum(row.data[0])
            # print(s)
            if s > 0:
                self.transitionMatrix[r] = row / s

        for r in range(self.transitionMatrix.shape[0]):
            vstr = str(r)+": "
            for c in range(self.transitionMatrix.shape[1]):
                if self.transitionMatrix[r, c] == 0.0:
                    vstr += "    "
                else:
                    vstr += str(self.transitionMatrix[r, c]) + " "
            print(vstr)

        # edges = list()
        # for s in self.prefixtree.states.values():
        #     for e in self.prefixtree.get_children(s):
        #         edges.append((s.id, e.id, s.properties.log_templates, e.properties.log_templates))
        #
        # edges.sort()
        # [print(e) for e in edges]
        # for e in self.prefixtree.edges.list:
        #     self.prefixtree.edges.get_values()
        # print(self.prefixtree.edges.list)
        # print(self.prefixtree.state_indices_by_id.values())
        #
        # print(self.log_templates)
        # print(self.terminal_index)

    def find_duplicates(self, index, threshold: float = 0.5) -> List[int]:
        # result = list()

        # found = False
        # for a in result:
        #     if index in a:
        #         found = True
        #         break
        # if not found:
        result = [index]
        row = self.transitionMatrix.getrow(index)
        for a in range(self.transitionMatrix.shape[0]):
            if a != index:
                row2 = self.transitionMatrix.getrow(a)
                all_rows = set(row.rows[0] + row2.rows[0]) # 2-dimensional, but we only want the first one
                equivalent = True
                for pos in all_rows:
                    if abs(row[0, pos] - row2[0, pos]) > threshold:
                        equivalent = False
                        break
                if equivalent:
                    result.append(a)

        return result if len(result) > 1 else []



    def do_it(self):
        # self.train("../../tests/resources/testlogs/")
        # self.train("../../resources/traces/")
        self.train("../../../all")
        print('Searching for duplicates')
        print(len(self.log_templates))
        print(self.log_templates)
        a_pool = multiprocessing.Pool()
        print(a_pool.map(self.find_duplicates, range(self.transitionMatrix.shape[0])))
        # dups = [a for a in a_pool.map(self.find_duplicates, range(self.transitionMatrix.shape[0])) if len(a) > 0]
        # print(self.transitionMatrix.getrow(dups[0][0]))
        # print(self.transitionMatrix.getrow(dups[0][5]))
        # print(dups[0])
        # print(a_pool.map(self.find_duplicates, range(self.transitionMatrix.shape[0])))




if __name__ == '__main__':
    # chain = MarkovChain(MarkovChain.ALL_POSSIBLE_LOOPS)
    # chain = MarkovChain(MarkovChain.SINGULAR_LOOPS)
    chain = MarkovChain(MarkovChain.ONLY_PROBABILISTIC)
    # todo merge probability of 1
    # todo search for
    chain.do_it()
