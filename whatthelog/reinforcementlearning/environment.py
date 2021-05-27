import random

from gym import Env
from gym.spaces import Discrete

from whatthelog.definitions import PROJECT_ROOT
from whatthelog.prefixtree.evaluator import Evaluator
from whatthelog.prefixtree.graph import Graph
from whatthelog.prefixtree.state import State
from whatthelog.reinforcementlearning.actionspace import ActionSpace, Actions
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


class GraphEnv(Env):
    MAX_OUTGOING_EDGES = 2

    def __init__(self, graph: Graph, syntax_tree: SyntaxTree):
        self.graph = graph
        self.stack = list()
        self.state_mapping = {}
        for i in [0, 1]:
            for j in [0, 1, 2, 3]:
                self.state_mapping[str(i) + str(j)] = len(self.state_mapping)

        self.visited = set()

        self.evaluator = Evaluator(graph, syntax_tree,
                                   PROJECT_ROOT.joinpath("resources/traces"),
                                   PROJECT_ROOT.joinpath(
                                       "resources/negative_traces"))
        self.syntax_tree = syntax_tree

        self.action_space = ActionSpace(Actions)

        # States: (if they are the same log template, number of outgoing states max MAX_OUTGOING_EDGES)
        self.observation_space = Discrete(2 * self.MAX_OUTGOING_EDGES + 2 + 2)

        self.state = self.graph.start_node
        self.stack.append(self.state)
        self.decode(self.state)

    def step(self, action):
        self.visited.add(self.state)
        outgoing = self.graph.get_outgoing_states_not_self(self.state)
        invalid_action = False
        new_outgoing = []

        if action == Actions.DONT_MERGE.value:
            self.state = self.stack.pop()

        elif action == Actions.MERGE_FIRST.value:
            if len(outgoing) == 2:
                self.graph.merge_states(self.state, outgoing[0])
            else:
                invalid_action = True

        elif action == Actions.MERGE_SECOND.value:
            if len(outgoing) == 2:
                self.graph.merge_states(self.state, outgoing[1])
            else:
                invalid_action = True

        elif action == Actions.MERGE_ALL:
            outgoing = self.graph.get_outgoing_states_not_self(self.state)
            for outgoing_state in outgoing:
                self.graph.merge_states(self.state, outgoing_state)

        new_outgoing = [s for s in self.graph.get_outgoing_states_not_self(self.state) if s not in self.visited]
        self.stack += new_outgoing

        if invalid_action is True:
            reward = 0
        else:
            reward = self.__get_reward()

        if len(self.stack) == 0:
            done = True
        else:
            done = False

        info = {}

        return self.decode(self.state), reward, done, info

    def reset(self):
        self.state = self.graph.start_node
        return self.decode(self.state)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def __get_reward(self):
        return self.evaluator.evaluate(0.5, 0.5)

    def decode(self, state: State) -> int:
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

        value = str(int(equivalent)) + str(len(outgoing))

        if value not in self.state_mapping:
            print("NEW VALUE: ", value)
            self.state_mapping[value] = len(self.state_mapping)

        return self.state_mapping[value]

    def get_valid_actions(self):
        if self.state.is_terminal is True:
            return [Actions.DONT_MERGE.value]
        if self.state == self.graph.start_node:
            return [Actions.DONT_MERGE.value]
        outgoing = self.graph.get_outgoing_states_not_self(self.state)
        return self.action_space.get_valid_actions(len(outgoing))
