# -*- coding: utf-8 -*-
"""
Created on Tuesday 04/20/2021
Author: Tommaso Brandirali
Email: tommaso.brandirali@gmail.com
"""

#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Union
import re

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.auto_printer import AutoPrinter

#****************************************************************************************************
# Prefix Tree
#****************************************************************************************************

@dataclass
class PrefixTree(AutoPrinter):
    """
    A data class representing the prefix tree for a system.
    Instances of this class should be created using the
    """

    name: str
    prefix: str
    isRegex: bool
    __children: List[PrefixTree] = field(default_factory=lambda: [])

    #================================================================================
    # Class Methods
    #================================================================================

    def get_children(self):
        """
        Prefix tree children getter.
        """
        return self.__children

    def insert(self, child: PrefixTree) -> None:
        """
        Add new child to tree.
        """

        self.__children.append(child)

    def search(self, input: str) -> Union[PrefixTree, None]:
        """
        Recursively search the prefix tree for Nodes matching the
        :param input: the string to match
        :return: the prefix tree node representing the best match, or None if no match found
        """

        prefix = re.escape(self.prefix) if not self.isRegex else self.prefix
        pattern = re.compile(prefix)
        stem = re.sub(pattern, '', input, 1)

        # Prefix match found
        if stem != input:

            for child in self.__children:

                result = child.search(stem)

                # Child match found
                if result:
                    return result

            return self

        return None
