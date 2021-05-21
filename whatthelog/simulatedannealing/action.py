import abc

from whatthelog.prefixtree.graph import Graph
from whatthelog.prefixtree.state import State


class Action(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def perform(*args, **kwargs):
        pass


class MergeRandomState(Action):

    @staticmethod
    def perform(graph: Graph):
        state: State = graph.get_random_state()
        other_state: State = graph.get_random_child(state)
        graph.full_merge_states(state, other_state)
