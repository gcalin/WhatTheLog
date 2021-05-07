#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import annotations
from ast import literal_eval
from dataclasses import dataclass, field
from typing import List


#****************************************************************************************************
# Edge Properties
#****************************************************************************************************

@dataclass
class EdgeProperties:

    props: List[str] = field(default_factory=lambda: [])

    def __str__(self):
        return str(self.props)

    @staticmethod
    def parse(string: str):
        return EdgeProperties(literal_eval(string))