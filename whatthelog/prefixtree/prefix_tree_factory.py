#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
from pathlib import Path
import pickle
from tqdm import tqdm

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.state import State
from whatthelog.syntaxtree.parser import Parser
from whatthelog.exceptions import UnidentifiedLogException
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.auto_printer import AutoPrinter

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Global Variables
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def print(msg): AutoPrinter.static_print(msg)


#****************************************************************************************************
# Prefix Tree Factory
#****************************************************************************************************

class PrefixTreeFactory(AutoPrinter):
    """
    Prefix tree factory class.
    """

    @staticmethod
    def get_prefix_tree(traces_dir: str, config_file_path: str) -> PrefixTree:
        """
        Parses a full tree from a set of log traces in a common directory,
        using a user-supplied syntax tree from an input configuration file.
        Assumes that all log traces are linear.
        :param traces_dir: the directory containing the log files to be parsed
        :param config_file_path: the configuration file describing the syntax tree
        :return: the full prefix tree
        """

        return PrefixTreeFactory.__generate_prefix_tree(traces_dir, config_file_path)

    @staticmethod
    def pickle_tree(tree: PrefixTree, file: str) -> None:
        """
        Pickles and dumps the given tree instance into a given file.
        If the file does not exist it will be created.
        :param tree: the tree instance to pickle
        :param file: the file to dump the pickled tree to
        """

        with open(file, 'wb+') as f:
            pickle.dump(tree, f)

    @staticmethod
    def unpickle_tree(file: str) -> PrefixTree:
        """
        Parses a pickled tree instance from a file.
        :param file: the pickle file representing the instance
        :return: the parsed PrefixTree instance
        """

        if not os.path.isfile(file):
            raise FileNotFoundError("Pickle file not found!")

        with open(file, 'rb') as f:
            tree = pickle.load(f)

        return tree

    @staticmethod
    def __generate_prefix_tree(log_dir: str, config_file: str) -> PrefixTree:
        """
        Script to parse log file into prefix tree.

        :param log_dir: Path to directory containing traces.
        :param config_file: Path to configuration file for syntax tree
        :return: Prefix tree along with a dictionary mapping log templates
         to unique ids.
        """

        if not os.path.isdir(log_dir):
            raise NotADirectoryError("Log directory not found!")
        if not os.path.isfile(config_file):
            raise FileNotFoundError("Config file not found!")

        print("Parsing syntax tree...")

        syntax_tree = Parser().parse_file(config_file)
        prefix_tree = PrefixTree(State([""]))

        print("Parsing traces...")

        pbar = tqdm(os.listdir(log_dir), file=sys.stdout, leave=False)
        for filename in pbar:
            filepath = str(Path(log_dir).joinpath(filename)).strip()
            PrefixTreeFactory.__parse_trace(filepath, syntax_tree, prefix_tree)

        return prefix_tree

    @staticmethod
    def __parse_trace(tracepath: str,
                      syntax_tree: SyntaxTree,
                      prefix_tree: PrefixTree) -> PrefixTree:
        """
        Function that parses a trace file and modifies the given
        prefix tree to include it.

        :param tracepath: The path to the trace file to parse
        :param syntax_tree: The syntax tree used to get the log template from the log
        :param prefix_tree: The current prefix tree to be used
        :return: None
        """

        parent = prefix_tree.get_root()
        nodes = prefix_tree.get_children(parent)

        with open(tracepath, 'r') as file:
            for log in file:

                template = syntax_tree.search(log).name
                if template is None:
                    raise UnidentifiedLogException(
                        log + " was not identified as a valid log.")

                exists = False

                for node in nodes:
                    if template in node.properties.log_templates:
                        parent = node
                        nodes = prefix_tree.get_children(parent)
                        exists = True
                        break

                if not exists:
                    child = State([template])
                    prefix_tree.add_child(child, parent)

                    parent = child
                    nodes = prefix_tree.get_children(child)

        return prefix_tree
