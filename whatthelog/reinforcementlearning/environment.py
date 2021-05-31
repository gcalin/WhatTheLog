import numpy as np
from gym import Env
from gym.spaces import Discrete

from whatthelog.prefixtree.evaluator import Evaluator
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.prefixtree.state import State
from whatthelog.reinforcementlearning.actionspace import ActionSpace, Actions
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


class GraphEnv(Env):
    MAX_OUTGOING_EDGES = 2
    """
    Constant defining the state space. Each key is a property of the
    state and its value is the possible combinations of this properties.
    For example EQUAL_TEMPLATES is a prop that has two options (true or false)
    """
    STATE_OPTIONS = {
        'EQUAL_TEMPLATES': 2,
        'OUTGOING_EDGES': MAX_OUTGOING_EDGES + 1 + 1
        # 1 for 0 edges and one for larger than
    }

    def __init__(self, prefix_tree_pickle_path: str, syntax_tree: SyntaxTree,
                 positive_traces_path: str, negative_traces_path: str):
        self.prefix_tree_pickle_path = prefix_tree_pickle_path
        self.positive_traces_path = positive_traces_path
        self.negative_traces_path = negative_traces_path

        self.graph = PrefixTreeFactory().unpickle_tree(
            self.prefix_tree_pickle_path)
        self.syntax_tree = syntax_tree
        self.evaluator = Evaluator(self.graph, syntax_tree,
                                   positive_traces_path, negative_traces_path)

        self.stack = list()
        self.state_mapping = {}
        self.__create_state_mapping()
        self.action_sequence = []

        self.visited = set()

        self.action_space = ActionSpace(Actions)

        # States: (if they are the same log template, number of outgoing states max MAX_OUTGOING_EDGES)
        self.observation_space = Discrete(
            np.product(list(self.STATE_OPTIONS.values())))

        self.state = self.graph.start_node
        self.outgoing = self.graph.get_outgoing_states_not_self(self.state)
        self.encode(self.state)

        for out in self.outgoing:
            self.stack.append(out)

    def __create_state_mapping(self):
        properties = list(self.STATE_OPTIONS.values())
        length = np.product(properties)

        # TODO: Generalize this
        for i in range(properties[0]):
            for j in range(properties[1]):
                self.state_mapping[str(i) + str(j)] = len(
                    self.state_mapping)

    def step(self, action: int):
        if self.is_valid_action(action) is False:
            return self.encode(self.state), 0, False, {}
        self.action_sequence.append(action)
        if len(self.stack) == 0:
            done = True
        else:
            done = False

        if action == Actions.DONT_MERGE.value:
            self.visited.add(self.state)
            if len(self.stack) != 0:
                next_node = self.get_next_node()
                if next_node is None:
                    return self.encode(self.state), 0, True, {}
                else:
                    self.state = next_node

        elif action == Actions.MERGE_ALL.value:
            self.state = self.graph.full_merge_states_all_children(self.state)
        elif action == Actions.MERGE_FIRST.value:
            self.state = self.graph.full_merge_states(self.state,
                                                      self.outgoing[0])
        elif action == Actions.MERGE_SECOND.value:
            self.state = self.graph.full_merge_states(self.state,
                                                      self.outgoing[1])

        self.outgoing = self.graph.get_outgoing_states_not_self(self.state)

        self.stack = list(
            filter(lambda x: x in self.graph and x not in self.visited,
                   self.stack))
        self.stack += list(
            filter(lambda x: x not in self.visited, self.outgoing))

        reward = self.__get_reward()

        info = {}

        return self.encode(self.state), reward, done, info

    def reset(self):
        self.graph = PrefixTreeFactory().unpickle_tree(
            self.prefix_tree_pickle_path)
        self.state = self.graph.start_node
        self.outgoing = self.graph.get_outgoing_states_not_self(self.state)
        self.evaluator = Evaluator(self.graph, self.syntax_tree,
                                   self.positive_traces_path,
                                   self.negative_traces_path)
        self.stack = [] + self.outgoing
        self.visited = set()
        self.action_sequence = []

        return self.encode(self.state)

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
        return self.evaluator.evaluate(0.5, 0.5)

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
        equivalent = True
        for outgoing_state in outgoing:
            if state.is_equivalent(outgoing_state) is False:
                equivalent = False

        value = str(int(equivalent)) + str(len(self.outgoing))

        if value not in self.state_mapping:
            print("NEW VALUE: ", value)
            self.state_mapping[value] = len(self.state_mapping)

        return self.state_mapping[value]

    def get_valid_actions(self):
        if self.state.is_terminal is True:
            return [Actions.DONT_MERGE.value]
        if self.state == self.graph.start_node:
            return [Actions.DONT_MERGE.value]

        return self.action_space.get_valid_actions(len(self.outgoing))

    def is_valid_action(self, action: int) -> bool:
        if action == Actions.DONT_MERGE.value or action == Actions.MERGE_ALL.value:
            return True
        elif action == Actions.MERGE_FIRST.value or action == Actions.MERGE_SECOND.value:
            return len(self.outgoing) == 2
