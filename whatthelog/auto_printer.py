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
import os


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

    @staticmethod
    def static_print(text):
        frame = inspect.stack()[1]
        filename = frame[0].f_code.co_filename
        print(f"[ {os.path.basename(filename)} ] - {text}")

    @staticmethod
    def static_prefix():
        return f"[ {os.path.basename(inspect.stack()[1][0].f_code.co_filename)} ] - "