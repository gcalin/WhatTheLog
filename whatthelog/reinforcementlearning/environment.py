import copy
import json
from typing import List

import numpy as np
import scipy.stats
from gym import Env
from gym.spaces import Discrete

from whatthelog.definitions import PROJECT_ROOT
from whatthelog.prefixtree.evaluator import Evaluator
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.prefixtree.state import State
from whatthelog.reinforcementlearning.actionspace import ActionSpace, Actions
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


class GraphEnv(Env):
    MAX_OUTGOING_EDGES = 3

    """
    Constant defining the state space. Each key is a property of the
    state and its value is the possible combinations of this properties.
    For example EQUAL_TEMPLATES is a prop that has two options (true or false)
    """
    STATE_OPTIONS = {
        'OUTGOING_EDGES': MAX_OUTGOING_EDGES + 1 + 1,
        'ENTROPY_VALUES': 5,
    }

    def __init__(self, prefix_tree_pickle_path: str, syntax_tree: SyntaxTree,
                 positive_traces: List[List[str]],
                 negative_traces: List[List[str]]):
        self.prefix_tree_pickle_path = prefix_tree_pickle_path
        self.positive_traces = positive_traces
        self.negative_traces = negative_traces

        self.graph = PrefixTreeFactory().unpickle_tree(
            self.prefix_tree_pickle_path)
        self.syntax_tree = syntax_tree
        self.evaluator = Evaluator(self.graph, syntax_tree, positive_traces,
                                   negative_traces)

        self.stack = list()
        self.state_mapping = {}
        self.__create_state_mapping()

        with open(PROJECT_ROOT.joinpath("resources/states.json"), 'w+') as f:
            json.dump(self.state_mapping, f, indent=2)
        self.visited = set()

        self.first_step = True

        self.action_space = ActionSpace(Actions)

        # States: (if they are the same log template, number of outgoing states max MAX_OUTGOING_EDGES)
        self.observation_space = Discrete(
            np.product(list(self.STATE_OPTIONS.values())))
        self.current_node = None
        self.outgoing = None
        self.frequencies = None
        self.entropy_d = None
        self.__set_current_node(self.graph.start_node)
        self.__encode_current_node()
        self.previous_accuracy = 0
        self.previous_size = 0

        self.checkpoint_tree = copy.deepcopy(self.graph)
        self.checkpoint_current_node = self.current_node.state_id

        self.step_count = 0

        for out in self.outgoing:
            self.stack.append(out)

    def __set_current_node(self, node: State):
        self.current_node = node
        outgoing = self.graph.get_outgoing_states_with_edges_no_self_no_terminal(
            node)
        self.outgoing = [out[0] for out in outgoing]

        passes = [out[1].props for out in outgoing]
        total_passes = np.sum(passes)
        self.frequencies = [value / total_passes for value in passes]

        if len(self.frequencies) <= 1:
            entropy = scipy.stats.entropy(self.frequencies)
        else:
            entropy = 1 / np.log2(len(self.frequencies)) * scipy.stats.entropy(
                self.frequencies)
        self.entropy_d = GraphEnv.round(entropy)

    def __create_state_mapping(self):
        properties = list(self.STATE_OPTIONS.values())

        # TODO: Generalize this
        for i in range(properties[0]):
            for j in range(properties[1]):
                self.state_mapping[str(i) + str(j)] = len(self.state_mapping)

    def step(self, action: int):
        self.step_count += 1
        info = {}
        done = False

        if len(self.stack) == 0:
            done = True
            info["stop"] = True
        else:
            info["stop"] = False

        # self.checkpoint_tree = copy.deepcopy(self.graph)
        # self.checkpoint_current_node = self.current_node.state_id

        if action == Actions.DONT_MERGE.value or self.current_node in self.visited:
            self.visited.add(self.current_node)
            if len(self.stack) != 0:
                next_node = self.get_next_node()
                if next_node is None:
                    self.current_node = next_node
                    return self.__encode_current_node(), 0, True, {}
                else:
                    self.current_node = next_node
        else:
            if action == Actions.MERGE_ALL.value:
                self.current_node = self.graph.full_merge_states_with_children(
                    self.current_node, self.visited)
            elif action == Actions.MERGE_FIRST.value:
                index = np.argmax(self.frequencies)
                self.current_node = self.graph.full_merge_states(
                    self.current_node,
                    self.outgoing[index], self.visited)
            elif action == Actions.MERGE_SECOND.value:
                index = np.argsort(self.frequencies, axis=0)[-2]
                self.current_node = self.graph.full_merge_states(
                    self.current_node,
                    self.outgoing[index], self.visited)
            elif action == Actions.MERGE_THIRD.value:
                index = np.argsort(self.frequencies, axis=0)[-3]
                self.current_node = self.graph.full_merge_states(
                    self.current_node,
                    self.outgoing[index], self.visited)
            elif action == Actions.MERGE_FIRST_TWO.value:
                indexes = np.argsort(self.frequencies, axis=0)[-2:]
                self.current_node = self.graph.full_merge_states_with_children(
                    self.current_node, self.visited, indexes)
            elif action == Actions.MERGE_LAST_TWO.value:
                indexes = np.argsort(self.frequencies, axis=0)[:2]
                self.current_node = self.graph.full_merge_states_with_children(
                    self.current_node, self.visited, indexes)
            elif action == Actions.MERGE_FIRST_AND_LAST.value:
                indexes = np.argsort(self.frequencies, axis=0)
                indexes = [indexes[0], indexes[-1]]
                self.current_node = self.graph.full_merge_states_with_children(
                    self.current_node, self.visited, indexes)

        self.__set_current_node(self.current_node)

        precision, recall, f1score, specificity, size = self.evaluator.evaluate()
        accuracy = 0.5 * f1score + 0.5 * specificity

        if self.first_step:
            reward = 0
            self.first_step = False
            self.previous_accuracy = accuracy
            self.previous_size = size
        else:
            if accuracy < 0.9:
                reward = accuracy - self.previous_accuracy
            elif accuracy < self.previous_accuracy:
                reward = size - self.previous_size
            else:
                if size > self.previous_size:
                    reward = (size - self.previous_size) * 10
                else:
                    reward = accuracy - self.previous_accuracy

        # reward = temp_reward - self.previous
        # self.previous = temp_reward

            info.update({
            "precision": precision,
            "recall": recall,
            "f1score": accuracy,
            "specificity": specificity,
            "size": size
        })
        if reward < 0:
            # self.graph = copy.deepcopy(self.checkpoint_tree)
            # self.current_node = self.graph.get_state_by_id(
            #     self.checkpoint_current_node)
            #
            # self.__set_current_node(self.current_node)
            # # self.outgoing = self.graph.get_outgoing_states_not_self(self.current_node)
            # self.evaluator.model = self.graph
            done = True
            info["stop"] = False
        else:
            self.previous_accuracy = accuracy
            self.previous_size = size

        self.stack = list(
            filter(lambda x: x in self.graph and x not in self.visited,
                   self.stack))
        self.stack += list(
            filter(lambda x: x not in self.visited, self.outgoing))

        # highest reward?


        return self.__encode_current_node(), reward, done, info

    def reset(self):
        self.previous_accuracy = 0
        self.previous_size = 0
        self.step_count = 0

        self.first_step = True

        self.graph = PrefixTreeFactory().unpickle_tree(
            self.prefix_tree_pickle_path)
        self.current_node = self.graph.start_node

        self.__set_current_node(self.current_node)
        # self.outgoing = self.graph.get_outgoing_states_not_self(self.current_node)
        self.evaluator = Evaluator(self.graph, self.syntax_tree,
                                   self.positive_traces,
                                   self.negative_traces)
        self.stack = [] + self.outgoing
        self.visited = set()

        return self.__encode_current_node()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def get_next_node(self):
        if len(self.stack) == 0:
            return None
        node = self.stack.pop()
        while node not in self.graph:
            try:
                self.stack.pop()
            except IndexError:
                return None
        return node

    def __encode_current_node(self) -> int:
        """
        Takes as input the state ie if the log templates are equivalent and the number of outgoing edges
        and returns the index of this state used in the q table.

        :param equivalent_log_template: Whether the root node for this step has the same log template
        as the child nodes
        :param outgoing_edges: Number of outgoing edges of this node
        :return: The index of this state
        """

        number_of_outgoing = len(self.outgoing)

        if number_of_outgoing > self.MAX_OUTGOING_EDGES:
            first_part = self.MAX_OUTGOING_EDGES + 1
        else:
            first_part = number_of_outgoing

        value = str(first_part) + str(self.entropy_d)

        return self.state_mapping[value]

    def get_valid_actions(self):
        if self.current_node.is_terminal is True:
            return [Actions.DONT_MERGE.value]
        if self.current_node == self.graph.start_node:
            return [Actions.DONT_MERGE.value]

        return self.action_space.get_valid_actions(len(self.outgoing),
                                                   self.entropy_d)

    @staticmethod
    def round(value: float):
        if 0 <= value < 0.2:
            return 0
        elif 0.2 <= value < 0.4:
            return 1
        elif 0.4 <= value < 0.6:
            return 2
        elif 0.6 <= value < 0.8:
            return 3
        elif 0.8 <= value <= 1:
            return 4
