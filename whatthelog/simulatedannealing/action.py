import abc
from typing import List

from whatthelog.prefixtree.graph import Graph
from whatthelog.prefixtree.state import State


class Action(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def perform(*args, **kwargs):
        pass


class MergeIncoming(Action):

    @staticmethod
    def perform(graph: Graph, state: State):
        pass
        # states: List[State] = graph.get_incoming_states(state)
        #
        # for incoming in states:
        #     if incoming is state:
        #         continue
        #
        #     graph.merge_states(state, incoming)
        #     graph.
