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

import inspect


#****************************************************************************************************
# Auto Printer
#****************************************************************************************************

class AutoPrinter:
    """
    A utility class to provide smart printing to derived classes.
    Smart printing simply appends the caller class and function names to the printed string.
    To use this feature, derived classes should simply call `self.print()` instead of `print()`.
    """

    def print(self, text: str, level: int = 1) -> None:

        try:
            print("[ %s ][ %s() ] - %s" % (str(self.__class__.__name__), str(inspect.stack()[level][3]), text), flush=True)
        except:
            print("ERROR: Failed to print '%s'" % str(text), flush=True)
