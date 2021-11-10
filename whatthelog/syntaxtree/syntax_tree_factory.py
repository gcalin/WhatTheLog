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

import json

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.auto_printer import AutoPrinter
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


#****************************************************************************************************
# Syntax Tree Factory
#****************************************************************************************************

class SyntaxTreeFactory(AutoPrinter):
    """
    A factory class for parsing a configuration file into a compiled Prefix Tree.
    """

    def __init__(self, abbadingo_format: bool = False):
        self.abbadingo_format = abbadingo_format
        self.terminal_nodes = 0

    #================================================================================
    # Parse input
    #================================================================================

    def parse_file(self, filepath: str) -> SyntaxTree:
        """
        Parses a configuration file in JSON format to a compiled Prefix Tree.
        The input JSON is expected to be a dictionary, with each nested node in the following format:
        {
            name: string
            prefix: string (possibly regex pattern)
            children: [...]
        }.
        :param filepath: the path to the configuration JSON file
        :return: a compiled instance of PrefixTree
        """

        with open(filepath, 'r') as f:
            configs = json.load(f)
            assert isinstance(configs, dict), f"Invalid configuration file: expected 'dict' but got '{configs.__class__}'"
            return self.__parse(configs)

    def __parse(self, configs: dict) -> SyntaxTree:

        name = configs["name"]
        prefix = configs["syntax"]
        isRegex = configs["isRegex"]
        children = configs["children"]

        tree = SyntaxTree(name, prefix, isRegex)
        for child in children:
            tree.insert(self.__parse(child))

        # Add the number corresponding to the abbadingo format
        if len(children) == 0 and self.abbadingo_format:
            tree.index = self.terminal_nodes
            self.terminal_nodes += 1

        return tree
