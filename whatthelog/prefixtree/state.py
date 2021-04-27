from dataclasses import dataclass
from re import Pattern
from typing import List

@dataclass
class State:

    """
    Class representing a state. Holds a list of all log template ids
    represented by this state.
    """

    log_ids: List[Pattern]

    def __str__(self):
        if len(self.log_ids) == 1:
            return str(self.log_ids[0])

        return str(self.log_ids)

    def __repr__(self):
        return self.__str__()
