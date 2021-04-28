import os
from typing import List, Dict, Tuple

from whatthelog.prefixtree.prefix_tree_graph import PrefixTreeGraph
from whatthelog.prefixtree.state import State
from whatthelog.syntaxtree.parser import Parser
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


class PrefixTreeFactory:
    """
    Prefix tree factory.
    """
    @staticmethod
    def get_prefix_tree(traces_dir: str, config_file_path: str) -> PrefixTreeGraph:
        prefix_tree = PrefixTreeFactory.__generate_prefix_tree(traces_dir, config_file_path)
        return prefix_tree

    @staticmethod
    def __generate_prefix_tree(log_dir: str, config_file: str) -> PrefixTreeGraph:
        """
        Script to parse log file into prefix tree.

        :param log_dir: Path to directory containing traces.
        :param config_file: Path to configuration file for syntax tree
        :return: Prefix tree along with a dictionary mapping log templates
         to unique ids.
        """
        syntax_tree = Parser().parse_file(config_file)
        prefix_tree = PrefixTreeGraph(State([""]))

        for filename in os.listdir(log_dir):
            with open(log_dir + filename, 'r') as f:
                logs = [log.strip() for log in f.readlines()]

            PrefixTreeFactory.__parse_trace(logs, syntax_tree, prefix_tree)
        return prefix_tree

    @staticmethod
    def __parse_trace(logs: List[str],
                      syntax_tree: SyntaxTree,
                      prefix_tree: PrefixTreeGraph) -> PrefixTreeGraph:
        """
        Function that parses a trace file and modifies the given
        prefix tree to include it.

        :param logs: A list of logs representing this trace
        :param log_templates: The mapping between log templates and log ids
        :param syntax_tree: The syntax tree used to get the log template from
        the log
        :param prefix_tree: The current prefix tree to be used
        :return: None
        """
        parent = prefix_tree.get_root()
        nodes = prefix_tree.get_children(parent)

        for log in logs:

            template = syntax_tree.search(log).name
            if template is None:
                raise UnidentifiedLogException(
                    log + " was not identified as a valid log.")

            exists = False

            for node in nodes:
                if template in node.log_templates:
                    parent = node
                    nodes = prefix_tree.get_children(parent)
                    exists = True
                    continue

            if not exists:
                child = State([template])
                prefix_tree.add_child(child, parent)

                parent = child
                nodes = prefix_tree.get_children(child)

        return prefix_tree


class UnidentifiedLogException(Exception):
    pass
