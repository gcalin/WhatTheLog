#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import annotations
from ast import literal_eval
from dataclasses import dataclass, field

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.exceptions import InvalidPropertiesException


#****************************************************************************************************
# Edge Properties
#****************************************************************************************************

@dataclass
class EdgeProperties:

    passes: int = field(default=1)

    def __str__(self):
        return str(self.passes)

    @staticmethod
    def parse(string: str):

        passes = literal_eval(string)
        if not type(passes) is int:
            raise InvalidPropertiesException(f"Error while deserializing edge properties: "
                                             f"expected Integer but got {str(type(passes))}")

        return EdgeProperties(passes)