from enum import Enum

from gym import Env
from gym.spaces import Discrete, Tuple, Box
from whatthelog.prefixtree.graph import Graph
import random

class Actions(Enum):
    DontMerge = 0
    Merge = 1

class GraphEnv(Env):
    def __init__(self, graph: Graph, max_steps: int):
        self.graph = graph

        self.max_steps = 100
        self.step = 0
        # Actions (0: Dont merge, 1: merge)
        self.action_space = Discrete(2)

        # States: (if they are the same log template, number of incoming, number of incoming)
        self.observation_space = Tuple((Discrete(2), Discrete(graph.size()), Discrete(graph.size()), Discrete(graph.size()), Discrete(graph.size())))

        state1 = graph.states[random.randint(0, graph.size() - 1)]
        state2 = graph.states[random.randint(0, graph.size() - 1)]

        self.candidate_merge = (state1, state2)

        self.state = (state1.is_equivalent(state2), len(graph.get_outgoing_states(state1)),
                      len(graph.get_incoming_states(state1)),
                      len(graph.get_outgoing_states(state2)),
                      len(graph.get_incoming_states(state2)))

    def step(self, action):
        self.step += 1

        if action == Actions.Merge.value:
            self.graph.merge_states(*self.candidate_merge)
            reward =

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass
