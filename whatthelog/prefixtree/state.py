#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import annotations
from typing import List


#****************************************************************************************************
# State
#****************************************************************************************************

class State:

    """
    Class representing a state. Holds a list of all log template ids
    represented by this state.
    """
    def __init__(self, log_templates: List[str]):
        """
        State constructor.

        :param log_templates: The log template ids this state holds.
        """
        self.log_templates: List[str] = log_templates

    def is_equivalent(self, other: State) -> bool:
        """
        Checks if the input state represents is equivalent to this one,
        this property is defined as having the same log templates.
        :param other: the state to check for equivalence with the current instance.
        :return: True if the input state is equivalent to this one, False otherwise.
        """

        return set(self.log_templates) == set(other.log_templates)

    def __str__(self):
        if len(self.log_templates) == 1:
            return str(self.log_templates[0])

        return str(self.log_templates)

    def __repr__(self):
        return self.__str__()
