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
from dataclasses import dataclass
from typing import List

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.auto_printer import AutoPrinter

#****************************************************************************************************
# Prefix Tree
#****************************************************************************************************

@dataclass()
class PrefixTree(AutoPrinter):
    """
    A data class representing the prefix tree for a system.
    Instances of this class should be created using the
    """

    name: str
    prefix: str
    children: List[PrefixTree]
