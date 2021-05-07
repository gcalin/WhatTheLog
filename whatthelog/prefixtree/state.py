# ****************************************************************************************************
# Imports
# ****************************************************************************************************

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import annotations
from typing import List

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.state_properties import StateProperties


# ****************************************************************************************************
# State
# ****************************************************************************************************

class State:
    """
    Class representing a state. Holds a list of all log template ids
    represented by this state.
    """

    __slots__ = ['properties', 'is_terminal']

    def __init__(self, log_templates: List[str], is_terminal: bool = False):
        """
        State constructor.

        :param log_templates: The log template ids this state holds.
        """
        self.properties = StateProperties(log_templates)
        self.is_terminal = is_terminal

    def is_equivalent(self, other: State) -> bool:
        """
        Checks if the input state represents is equivalent to this one,
        this property is defined as having the same log templates.
        :param other: the state to check for equivalence with the current instance.
        :return: True if the input state is equivalent to this one, False otherwise.
        """

        return self.properties == other.properties

    def get_properties(self):
        return self.properties

    def __str__(self):
        if len(self.properties) == 1:
            return str(self.properties.log_templates[0])

        return str(self.properties)

    def __repr__(self):
        return self.__str__()
