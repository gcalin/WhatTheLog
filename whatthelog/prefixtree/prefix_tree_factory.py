# ****************************************************************************************************
# Imports
# ****************************************************************************************************

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
from pathlib import Path
import pickle
from typing import List

from tqdm import tqdm

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.edge_properties import EdgeProperties
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.state import State
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory
from whatthelog.exceptions import UnidentifiedLogException
from whatthelog.auto_printer import AutoPrinter


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Global Variables
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def print(msg): AutoPrinter.static_print(msg)

# ****************************************************************************************************
# Prefix Tree Factory
# ****************************************************************************************************


class PrefixTreeFactory(AutoPrinter):
    """
    Prefix tree factory class.
    """

    @staticmethod
    def get_prefix_tree(traces_dir: str, config_file_path: str, remove_trivial_loops: bool = False,
                        files: List[str] = None) -> PrefixTree:
        """
        Parses a full tree from a set of log traces in a common directory,
        using a user-supplied syntax tree from an input configuration file.
        Assumes that all log traces are linear.
        :param traces_dir: the directory containing the log files to be parsed
        :param config_file_path: the configuration file describing the syntax tree
        :param remove_trivial_loops: Indicates whether trivial loops (subsequent states) should be merged.
        :param files: if given, will use the filepaths instead of the input directory
        :return: the full prefix tree
        """

        if files:
            return PrefixTreeFactory.__generate_prefix_tree_from_files(files, config_file_path, remove_trivial_loops)
        else:
            return PrefixTreeFactory.__generate_prefix_tree(traces_dir, config_file_path, remove_trivial_loops)

    @staticmethod
    def pickle_tree(tree: PrefixTree, file: str) -> None:
        """
        Pickles and dumps the given tree instance into a given file.
        If the file does not exist it will be created.
        :param tree: the tree instance to pickle
        :param file: the file to dump the pickled tree to
        """

        # --- Delete state indices table to save space since it will be invalid after unpickling ---
        tree.state_indices_by_id = {}

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
            tree: PrefixTree = pickle.load(f)

        return tree

    @staticmethod
    def __generate_prefix_tree(log_dir: str, config_file: str, remove_trivial_loops: bool) -> PrefixTree:
        """
        Script to parse log file into prefix tree.

        :param log_dir: Path to directory containing traces.
        :param config_file: Path to configuration file for syntax tree
        :param remove_trivial_loops: Indicates whether trivial loops (subsequent states) should be merged.
        :return: Prefix tree along with a dictionary mapping log templates
         to unique ids.
        """

        if not os.path.isdir(log_dir):
            raise NotADirectoryError("Log directory not found!")
        if not os.path.isfile(config_file):
            raise FileNotFoundError("Config file not found!")

        print("Parsing syntax tree...")

        syntax_tree = SyntaxTreeFactory().parse_file(config_file)
        prefix_tree = PrefixTree(syntax_tree, State([""]))

        print("Parsing traces...")

        pbar = tqdm(os.listdir(log_dir), file=sys.stdout, leave=False)
        for filename in pbar:
            filepath = str(Path(log_dir).joinpath(filename)).strip()
            PrefixTreeFactory.__parse_trace(filepath, prefix_tree, remove_trivial_loops)

        return prefix_tree

    @staticmethod
    def __generate_prefix_tree_from_files(files: List[str], config_file: str, remove_trivial_loops: bool) -> PrefixTree:

        for filepath in files:
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"File {filepath} not found!")
        if not os.path.isfile(config_file):
            raise FileNotFoundError("Config file not found!")

        print("Parsing syntax tree...")

        syntax_tree = SyntaxTreeFactory().parse_file(config_file)
        prefix_tree = PrefixTree(syntax_tree, State([""]))

        print("Parsing traces...")

        pbar = tqdm(files, file=sys.stdout, leave=False)
        for filepath in pbar:
            PrefixTreeFactory.__parse_trace(filepath, prefix_tree, remove_trivial_loops)

        return prefix_tree

    @staticmethod
    def __parse_trace(tracepath: str, prefix_tree: PrefixTree, remove_trivial_loops: bool) -> PrefixTree:
        """
        Function that parses a trace file and modifies the given
        prefix tree to include it.

        :param tracepath: The path to the trace file to parse
        :param prefix_tree: The current prefix tree to be used
        :param remove_trivial_loops: Indicates whether trivial loops (subsequent states) should be merged.
        :return: PrefixTree
        """

        current = prefix_tree.get_root()
        curr_children = prefix_tree.get_children(current)

        with open(tracepath, 'r') as file:
            for log in file:

                tree = prefix_tree.syntax_tree.search(log)
                if tree is None:
                    raise UnidentifiedLogException(
                        log + " was not identified as a valid log.")
                template = tree.name

                exists = False

                if remove_trivial_loops and current.properties.log_templates[0] == template:
                    # There will only be 1 template per state initially
                    if current not in curr_children:
                        prefix_tree.add_edge(current, current, EdgeProperties())
                    else:
                        prefix_tree.update_edge(current, current)
                else:
                    for node in curr_children:
                        if template in node.properties.log_templates:
                            prefix_tree.update_edge(current, node)
                            current = node
                            curr_children = prefix_tree.get_children(current)
                            exists = True
                            break

                    if not exists:
                        child = State([template])
                        prefix_tree.add_child(child, current)

                        current = child
                        curr_children = prefix_tree.get_children(current)

        # for n in nodes:
        #     if n.properties.log_templates[0] == 'terminal':
        #         return prefix_tree
        # prefix_tree.add_child(State(["terminal"], True), parent)
        # return prefix_tree
        current.is_terminal = True
        return prefix_tree
