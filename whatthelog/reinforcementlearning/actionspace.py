import random
from enum import Enum
from typing import List

from gym import Space


class ActionSpace(Space):

    def __init__(self, actions: Enum):
        super().__init__()
        self.actions = [e.value for e in actions]
        self.n = len(actions)

    def sample(self):
        return random.randint(0, len(self.actions) - 1)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        else:
            return False
        return 0 <= as_int < self.n

    def get_valid_actions(self, outgoing_edges: int) -> List[int]:
        if outgoing_edges == 1:
            return self.actions[:2]
        elif outgoing_edges == 2:
            return self.actions[:4]
        elif outgoing_edges == 3:
            return self.actions
        else:
            return [Actions.DONT_MERGE.value]


class Actions(Enum):
    DONT_MERGE = 0
    MERGE_ALL = 1
    MERGE_FIRST = 2
    MERGE_SECOND = 3
    MERGE_THIRD = 4
    MERGE_FIRST_TWO = 5
    MERGE_LAST_TWO = 6
    MERGE_FIRST_AND_LAST = 7


