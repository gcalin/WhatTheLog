# ****************************************************************************************************
# Imports
# ****************************************************************************************************

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import pickle
import sys
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from whatthelog.auto_printer import AutoPrinter
from whatthelog.exceptions import UnidentifiedLogException
from whatthelog.prefixtree.edge_properties import EdgeProperties
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.state import State
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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
    def get_prefix_tree(traces_dir: str, config_file_path: str, unique_graph: bool = True,
                        remove_trivial_loops: bool = False) -> PrefixTree:
        """
        Parses a full tree from a set of log traces in a common directory,
        using a user-supplied syntax tree from an input configuration file.
        Assumes that all log traces are linear.
        :param traces_dir: the directory containing the log files to be parsed
        :param config_file_path: the configuration file describing the syntax tree
        :param remove_trivial_loops: Indicates whether trivial loops (subsequent states) should be merged.
        :return: the full prefix tree
        """

        return PrefixTreeFactory.__generate_prefix_tree(traces_dir,
                                                        config_file_path, unique_graph,
                                                        remove_trivial_loops)

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
    def __generate_prefix_tree(log_dir: str, config_file: str, unique_graph: bool,
                               remove_trivial_loops: bool) -> PrefixTree:
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
        prefix_tree = PrefixTree(State([""]), State(["end"], is_terminal=True))

        print("Parsing traces...")

        pbar = tqdm(os.listdir(log_dir), file=sys.stdout, leave=False)
        mapping = {}
        for filename in pbar:
            filepath = str(Path(log_dir).joinpath(filename)).strip()
            if unique_graph:
                PrefixTreeFactory.parse_trace_unique_nodes(filepath, syntax_tree, prefix_tree,
                                                           mapping)
            else:
                PrefixTreeFactory.__parse_trace(filepath, syntax_tree, prefix_tree,
                                                           remove_trivial_loops)

        return prefix_tree

    @staticmethod
    def __parse_trace(tracepath: str,
                      syntax_tree: SyntaxTree,
                      prefix_tree: PrefixTree,
                      remove_trivial_loops: bool) -> PrefixTree:
        """
        Function that parses a trace file and modifies the given
        prefix tree to include it.

        :param tracepath: The path to the trace file to parse
        :param syntax_tree: The syntax tree used to get the log template from the log
        :param prefix_tree: The current prefix tree to be used
        :param remove_trivial_loops: Indicates whether trivial loops (subsequent states) should be merged.
        :return: PrefixTree
        """

        parent = prefix_tree.get_root()
        nodes = prefix_tree.get_children(parent)

        with open(tracepath, 'r') as file:
            for log in file:

                tree = syntax_tree.search(log)
                if tree is None:
                    raise UnidentifiedLogException(
                        log + " was not identified as a valid log.")
                template = tree.name

                exists = False

                if remove_trivial_loops and parent.properties.log_templates[
                    0] == template:
                    # There will only be 1 template per state initially
                    if parent not in nodes:
                        prefix_tree.add_edge(parent, parent, EdgeProperties())
                    else:
                        prefix_tree.update_edge(parent, parent)
                else:
                    for node in nodes:
                        if template in node.properties.log_templates:
                            prefix_tree.update_edge(parent, node)
                            parent = node
                            nodes = prefix_tree.get_children(parent)
                            exists = True
                            break

                    if not exists:
                        child = State([template])
                        prefix_tree.add_child(child, parent)

                        parent = child
                        nodes = prefix_tree.get_children(parent)

        if remove_trivial_loops:
            for n in nodes:
                if n.properties.log_templates[0] == 'terminal':
                    return prefix_tree
        prefix_tree.add_edge(parent, prefix_tree.get_terminal(),
                             EdgeProperties())
        return prefix_tree

    @staticmethod
    def parse_trace_unique_nodes(tracepath: str,
                                 syntax_tree: SyntaxTree,
                                 prefix_tree: PrefixTree,
                                 node_to_template_map: Dict[str, State]):
        parent = prefix_tree.get_root()
        nodes = prefix_tree.get_children(parent)

        with open(tracepath, 'r') as file:
            for log in file:

                tree = syntax_tree.search(log)
                if tree is None:
                    raise UnidentifiedLogException(
                        log + " was not identified as a valid log.")
                template = tree.name

                exists = False

                for node in nodes:
                    if template in node.properties.log_templates:
                        prefix_tree.update_edge(parent, node)
                        parent = node
                        nodes = prefix_tree.get_children(parent)
                        exists = True
                        break

                if not exists:
                    if template in node_to_template_map:
                        prefix_tree.add_edge(parent, node_to_template_map[template])
                        parent = node_to_template_map[template]
                        nodes = prefix_tree.get_children(parent)
                    else:
                        child = State([template])
                        node_to_template_map[template] = child

                        prefix_tree.add_child(child, parent)
                        parent = child
                        nodes = []

        prefix_tree.add_edge(parent, prefix_tree.get_terminal(),
                             EdgeProperties())
        return prefix_tree

