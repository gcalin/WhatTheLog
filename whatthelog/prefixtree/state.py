from typing import List


class State:
    """
    Class representing a state. Holds a list of all log template ids
    represented by this state.
    """
    def __init__(self, log_ids: List[int]):
        """
        State constructor.

        :param log_ids: The log template ids this state holds.
        """
        self.log_ids = log_ids

    def __str__(self):
        if len(self.log_ids) == 1:
            return str(self.log_ids[0])

        return str(self.log_ids)

    def __repr__(self):
        return self.__str__()
