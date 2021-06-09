import json
from typing import List

import copy
import numpy as np
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
    MAX_INCOMING_EDGES = 3

    """
    Constant defining the state space. Each key is a property of the
    state and its value is the possible combinations of this properties.
    For example EQUAL_TEMPLATES is a prop that has two options (true or false)
    """
    STATE_OPTIONS = {
        'OUTGOING_EDGES': MAX_OUTGOING_EDGES + 1 + 1,
        'INCOMING_EDGES': MAX_INCOMING_EDGES + 1 + 1,
        # 'FREQUENCY_FIRST': 3,
        # 'FREQUENCY_SECOND': 3,
        # 'FREQUENCY_THIRD': 3
    }

    def __init__(self, prefix_tree_pickle_path: str, syntax_tree: SyntaxTree,
                 positive_traces: List[List[str]], negative_traces: List[List[str]], w_accuracy: float, w_size: float):
        self.prefix_tree_pickle_path = prefix_tree_pickle_path
        self.positive_traces = positive_traces
        self.negative_traces = negative_traces
        self.w_accuracy = w_accuracy
        self.w_size = w_size
        self.graph = PrefixTreeFactory().unpickle_tree(self.prefix_tree_pickle_path)
        self.syntax_tree = syntax_tree
        self.evaluator = Evaluator(self.graph, syntax_tree, positive_traces, negative_traces)

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

        self.current_node = self.graph.start_node
        self.outgoing = self.graph.get_outgoing_states_not_self(self.current_node)
        self.encode(self.current_node)
        self.previous = 0

        self.previous_tree = copy.deepcopy(self.graph)
        self.previous_current_node = self.current_node.state_id

        for out in self.outgoing:
            self.stack.append(out)

    def __create_state_mapping(self):
        properties = list(self.STATE_OPTIONS.values())
        length = np.product(properties)

        # TODO: Generalize this
        for i in range(properties[0]):
            for j in range(properties[1]):
            #     for k in range(properties[2]):
            #         for l in range(properties[3]):
                self.state_mapping[str(i) + str(j)] = len(self.state_mapping)

    def step(self, action: int):
        if len(self.stack) == 0:
            done = True
        else:
            done = False

        self.previous_tree = copy.deepcopy(self.graph)
        self.previous_current_node = self.current_node.state_id

        if action == Actions.DONT_MERGE.value or self.current_node in self.visited:
            self.visited.add(self.current_node)
            if len(self.stack) != 0:
                next_node = self.get_next_node()
                if next_node is None:
                    return self.encode(self.current_node), 0, True, {}
                else:
                    self.current_node = next_node
        else:

            if action == Actions.MERGE_ALL.value:
                self.current_node = self.graph.full_merge_states_with_children(self.current_node, self.visited)
            if action == Actions.MERGE_FIRST.value:
                self.current_node = self.graph.full_merge_states(self.current_node,
                                                                 self.outgoing[0], self.visited)
            if action == Actions.MERGE_SECOND.value:
                self.current_node = self.graph.full_merge_states(self.current_node,
                                                                 self.outgoing[1], self.visited)
            if action == Actions.MERGE_THIRD.value:
                self.current_node = self.graph.full_merge_states(self.current_node,
                                                                 self.outgoing[2], self.visited)
            if action == Actions.MERGE_FIRST_TWO.value:
                self.current_node = self.graph.full_merge_states_with_children(self.current_node, self.visited, [0, 1])
            if action == Actions.MERGE_LAST_TWO.value:
                self.current_node = self.graph.full_merge_states_with_children(self.current_node, self.visited, [len(self.outgoing) - 2, len(self.outgoing) - 1])
            if action == Actions.MERGE_FIRST_AND_LAST.value:
                self.current_node = self.graph.full_merge_states_with_children(self.current_node, self.visited, [0, len(self.outgoing) - 1])

        self.outgoing = self.graph.get_outgoing_states_not_self(self.current_node)

        self.stack = list(
            filter(lambda x: x in self.graph and x not in self.visited,
                   self.stack))
        self.stack += list(
            filter(lambda x: x not in self.visited, self.outgoing))

        temp_reward = self.__get_reward()

        if self.first_step:
            reward = 0
            self.first_step = False
        else:
            reward = temp_reward - self.previous
        self.previous = temp_reward

        if reward < 0:
            done = True

        # highest reward?

        info = {}

        return self.encode(self.current_node), reward, done, info

    def reset(self):
        self.previous = 0
        self.first_step = True

        if len(self.stack) == 0:
            print("Start over")
            self.graph = PrefixTreeFactory().unpickle_tree(self.prefix_tree_pickle_path)
            self.current_node = self.graph.start_node

        else:
            print("Revert merge")
            self.graph = copy.deepcopy(self.previous_tree)
            self.current_node = self.graph.get_state_by_id(self.previous_current_node)


        self.outgoing = self.graph.get_outgoing_states_not_self(self.current_node)
        self.evaluator = Evaluator(self.graph, self.syntax_tree,
                                   self.positive_traces,
                                   self.negative_traces)
        self.stack = [] + self.outgoing
        self.visited = set()

        return self.encode(self.current_node)

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

    def __get_reward(self):
        return self.evaluator.evaluate(self.w_accuracy, self.w_size)

    def encode(self, state: State) -> int:
        """
        Takes as input the state ie if the log templates are equivalent and the number of outgoing edges
        and returns the index of this state used in the q table.

        :param equivalent_log_template: Whether the root node for this step has the same log template
        as the child nodes
        :param outgoing_edges: Number of outgoing edges of this node
        :return: The index of this state
        """
        outgoing = self.graph.get_outgoing_states_not_self(state)
        incoming = self.graph.get_incoming_states_not_self(state)

        # passes = list(map(lambda x: x[1].props, outgoing))
        #
        # total_passes = sum(passes)
        # frequencies = list(map(lambda x: self.round(x / total_passes), passes))
        # if len(frequencies) > 3:
        #     frequencies = []
        # while len(frequencies) < 3:
        #     frequencies.append(0)

        number_of_outgoing = len(outgoing)
        number_of_incoming = len(incoming)

        if number_of_outgoing > self.MAX_OUTGOING_EDGES:
            first_part = self.MAX_OUTGOING_EDGES + 1
        else:
            first_part = number_of_outgoing

        if number_of_incoming > self.MAX_INCOMING_EDGES:
            second_part = self.MAX_INCOMING_EDGES + 1
        else:
            second_part = number_of_incoming

        value = str(first_part) + str(second_part)

        # for f in frequencies:
        #     value += str(f)

        return self.state_mapping[value]

    def get_valid_actions(self):
        if self.graph.terminal_node in self.outgoing:
            if len(self.outgoing) == 1 and self.outgoing[0].is_terminal:
                return [Actions.DONT_MERGE.value]
            elif len(self.outgoing) == 2:
                if self.outgoing[0].is_terminal:
                    return [Actions.DONT_MERGE.value, Actions.MERGE_SECOND.value]
                elif self.outgoing[1].is_terminal:
                    return [Actions.DONT_MERGE.value, Actions.MERGE_FIRST.value]
            elif len(self.outgoing) == 3:
                if self.outgoing[0].is_terminal:
                    return [Actions.DONT_MERGE.value, Actions.MERGE_SECOND.value, Actions.MERGE_THIRD.value, Actions.MERGE_LAST_TWO.value]
                elif self.outgoing[1].is_terminal:
                    return [Actions.DONT_MERGE.value, Actions.MERGE_FIRST.value, Actions.MERGE_THIRD.value, Actions.MERGE_FIRST_AND_LAST.value]
                elif self.outgoing[2].is_terminal:
                    return [Actions.DONT_MERGE.value, Actions.MERGE_FIRST.value, Actions.MERGE_SECOND.value, Actions.MERGE_FIRST_TWO.value]

        if self.current_node.is_terminal is True:
            return [Actions.DONT_MERGE.value]
        if self.current_node == self.graph.start_node:
            return [Actions.DONT_MERGE.value]

        return self.action_space.get_valid_actions(len(self.outgoing))

    @staticmethod
    def round(frequency):
        if 0 <= frequency <= 0.2:
            return 0
        elif 0.8 <= frequency <= 1:
            return 2
        else:
            return 1
