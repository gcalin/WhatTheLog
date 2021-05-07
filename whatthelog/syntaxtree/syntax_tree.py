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
# Syntax Tree
#****************************************************************************************************

@dataclass
class SyntaxTree(AutoPrinter):
    """
    A data class representing the syntax tree for a system.
    Instances of this class should be created using the
    """

    name: str
    prefix: str
    isRegex: bool
    __children: List[SyntaxTree] = field(default_factory=lambda: [])

    def __post_init__(self):
        try:
            prefix = re.escape(self.prefix) if not self.isRegex else self.prefix
            self.__pattern = re.compile(prefix)
        except re.error:
            self.print(f"ERROR: Invalid pattern given for Node '{self.name}'")
            raise ValueError

    #================================================================================
    # Class Methods
    #================================================================================

    def get_children(self):
        """
        Syntax tree children getter.
        """
        return self.__children

    def get_pattern(self):
        """
        Syntax tree pattern getter.
        """
        return self.__pattern

    def insert(self, child: SyntaxTree) -> None:
        """
        Add new child to tree.
        """

        self.__children.append(child)

    def search(self, input: str) -> Union[SyntaxTree, None]:
        """
        Recursively search the prefix tree for Nodes matching the
        :param input: the string to match
        :return: the prefix tree node representing the best match, or None if no match found
        """

        # Prefix match found at the beginning of the string
        position = re.search(self.__pattern, input)
        if position and position.start() == 0:

            stem = re.sub(self.__pattern, '', input, 1)

            if len(self.__children) == 0:
                return self

            for child in self.__children:

                result = child.search(stem)

                # Child match found
                if result:
                    return result

        return None
