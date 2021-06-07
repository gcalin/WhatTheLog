
#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import annotations
from typing import List

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.state_properties import StateProperties


#****************************************************************************************************
# State
#****************************************************************************************************

class State:
    """
    Class representing a state. Holds a list of all log template ids
    represented by this state.
    """

    __slots__ = ['properties', 'is_terminal', 'state_id']

    def __init__(self,
                 single_template: List[str] = None,
                 is_terminal: bool = False,
                 state_id: int = -1,
                 log_templates: List[List[str]] = None):
        """
        State constructor.

        :param log_templates: The log template ids this state holds.
        """

        # This is a hacky solution; no time to refactor
        assert single_template is not None or log_templates is not None, "Some template must be provided"
        self.properties = StateProperties([single_template] if single_template else log_templates)
        self.is_terminal = is_terminal
        self.state_id = state_id

    def is_equivalent(self, other: State) -> bool:
        """
        Checks if the input state represents is equivalent to this one,
        this property is defined as having the same log templates.
        :param other: the state to check for equivalence with the current instance.
        :return: True if the input state is equivalent to this one, False otherwise.
        """

        return self.properties == other.properties

    def is_equivalent_weak(self, other: State) -> bool:
        """
        Checks if two states have any templates in common.
        """
        other_templates: List[List[str]] = other.get_properties().log_templates
        return any(map(lambda template: template in other_templates, self.get_properties().log_templates))

    def get_properties(self):
        return self.properties

    def __str__(self):
        if len(self.properties) == 1:
            return str(self.properties.log_templates[0])

        return str(self.properties)

    def __repr__(self):
        return self.__str__()
