# -*- coding: utf-8 -*-
"""
Created on Thursday 04/29/2021
Author: Tommaso Brandirali
Email: tommaso.brandirali@gmail.com
"""

#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from dataclasses import dataclass
from typing import List

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.auto_printer import AutoPrinter


#****************************************************************************************************
# State Properties
#****************************************************************************************************

@dataclass
class StateProperties(AutoPrinter):
    """
    Data class holding the properties of a state node.
    Multiple nodes can share the same properties, in that case they are considered 'equivalent.
    """

    __slots__ = ['log_templates']

    log_templates: List[str]

    def get_prop_hash(self) -> int:
        return hash(tuple(self.log_templates))

    def __hash__(self):
        return self.get_prop_hash()

    def __eq__(self, other):
        return set(self.log_templates) == set(other.log_templates)

    def __len__(self):
        return len(self.log_templates)

    def __str__(self):
        return str(self.log_templates)

    def __repr__(self):
        return str(self)
