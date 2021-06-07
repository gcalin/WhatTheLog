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
from typing import List, Dict, Tuple

from tqdm import tqdm

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.edge_properties import EdgeProperties
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.state import State
from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory
from whatthelog.exceptions import UnidentifiedLogException
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
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
    def get_prefix_tree(traces_dir: str,
                        config_file_path: str,
                        remove_trivial_loops: bool = False,
                        one_state_per_template: bool = False,
                        syntax_tree: SyntaxTree = None) -> PrefixTree:
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
                                                        config_file_path,
                                                        remove_trivial_loops,
                                                        one_state_per_template=one_state_per_template,
                                                        syntax_tree=syntax_tree)

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
    def __generate_prefix_tree(log_dir: str,
                               config_file: str,
                               remove_trivial_loops: bool,
                               one_state_per_template: bool = False,
                               syntax_tree: SyntaxTree = None) -> PrefixTree:
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
        if syntax_tree is None:
            syntax_tree = SyntaxTreeFactory().parse_file(config_file)

        prefix_tree = PrefixTree(State([""]))

        if one_state_per_template:
            terminal_state: State = State(["terminal"], True)
            prefix_tree.add_state(terminal_state)
            template_dict: Dict[str, State] = {"": prefix_tree.get_root(),
                                               "terminal": terminal_state}
        print("Parsing traces...")


        pbar = tqdm(os.listdir(log_dir), file=sys.stdout, leave=False)
        for filename in pbar:
            filepath = str(Path(log_dir).joinpath(filename)).strip()
            if one_state_per_template:

                PrefixTreeFactory.__parse_trace_compact(filepath, syntax_tree, prefix_tree, template_dict)
            else:
                PrefixTreeFactory.__parse_trace(filepath, syntax_tree, prefix_tree, remove_trivial_loops)

        return prefix_tree

    @staticmethod
    def parse_single_trace(tracepath: str,
                           syntax_tree: SyntaxTree,
                           remove_trivial_loops: bool = True) -> PrefixTree:
        return PrefixTreeFactory.__parse_trace(tracepath, syntax_tree, PrefixTree(State([""])), remove_trivial_loops)

    @staticmethod
    def parse_multiple_traces(traces_dir: str,
                              syntax_tree: SyntaxTree,
                              remove_trivial_loops: bool = True) -> List[PrefixTree]:

        res: List[PrefixTree] = []

        # Check if directory exists
        if not os.path.isdir(traces_dir):
            raise NotADirectoryError("Log directory not found!")

        for trace in os.listdir(traces_dir):
            res.append(PrefixTreeFactory.
                       parse_single_trace(os.path.join(traces_dir, trace), syntax_tree, remove_trivial_loops))

        return res

    @staticmethod
    def __parse_trace_compact(tracepath: str,
                              syntax_tree: SyntaxTree,
                              prefix_tree: PrefixTree,
                              template_dict: Dict[str, State]) -> Tuple[PrefixTree, Dict[str, State]]:
        """
        Function that parses a trace file and modifies the given
        prefix tree to include it.

        :param tracepath: The path to the trace file to parse
        :param syntax_tree: The syntax tree used to get the log template from the log
        :param prefix_tree: The current prefix tree to be used
        :return: PrefixTree
        """

        parent = prefix_tree.get_root()
        nodes = prefix_tree.get_children(parent)

        with open(tracepath, 'r') as file:
            x: int = 0
            for log in file:
                x += 1
                tree = syntax_tree.search(log)
                if tree is None:
                    raise UnidentifiedLogException(
                        log + " was not identified as a valid log.")
                template: str = tree.name
                if x == 9:
                    pass
                if template in template_dict:
                    exists = False
                    for node in nodes:
                        if template in node.properties.log_templates:
                            parent = node
                            nodes = prefix_tree.get_children(parent)
                            exists = True
                            break
                    if not exists:
                        prefix_tree.add_edge(parent, template_dict[template])
                        parent = template_dict[template]
                else:
                    child = State([template])
                    template_dict[template] = child
                    prefix_tree.add_child(child, parent)

                    parent = child
                    nodes = prefix_tree.get_children(parent)

        for n in nodes:
            if n.properties.log_templates[0] == 'terminal':
                return prefix_tree, template_dict
        prefix_tree.add_edge(parent, template_dict["terminal"])
        return prefix_tree, template_dict

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

                if remove_trivial_loops and parent.properties.log_templates[0] == [template]:
                    # There will only be 1 template per state initially
                    if parent not in nodes:
                        prefix_tree.add_edge(parent, parent, EdgeProperties([]))
                else:
                    for node in nodes:
                        if [template] in node.properties.log_templates:
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
                if n.properties.log_templates[0] == ['terminal']:
                    return prefix_tree
        prefix_tree.add_child(State(["terminal"], True), parent)
        return prefix_tree
