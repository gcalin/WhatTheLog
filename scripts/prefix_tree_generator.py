import os
from pprint import pprint
import sys
from typing import List, Dict, Tuple
from tqdm import tqdm

from whatthelog.exceptions.unidentified_log_exception import \
    UnidentifiedLogException
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.state import State
from whatthelog.prefixtree.visualizer import Visualizer
from whatthelog.syntaxtree.parser import Parser
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


def generate_prefix_tree(log_dir: str, config_file: str) -> \
        Tuple[PrefixTree, Dict[str, int]]:
    """
    Script to parse log file into prefix tree.

    :param log_dir: Path to directory containing traces.
    :param config_file: Path to configuration file for syntax tree
    :return: Prefix tree along with a dictionary mapping log templates
     to unique ids.
    """
    syntax_tree = Parser().parse_file(config_file)
    prefix_tree = PrefixTree(State([0]), None)
    log_templates = {"": 0}

    pbar = tqdm(total=len(os.listdir(log_dir)), file=sys.stdout, leave=False)
    for filename in os.listdir(log_dir):
        with open(log_dir + filename, 'r') as f:
            logs = [log.strip() for log in f.readlines()]

        parse_trace(logs, log_templates, syntax_tree, prefix_tree)
        pbar.update(1)
    return prefix_tree, log_templates


def parse_trace(logs: List[str],
                log_templates: Dict[str, int],
                syntax_tree: SyntaxTree,
                prefix_tree: PrefixTree) -> PrefixTree:
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
    parent = prefix_tree
    nodes = prefix_tree.get_children()

    for log in logs:
        template = syntax_tree.search(log).name
        if template is None:
            raise UnidentifiedLogException(
                log + " was not identified as a valid log.")

        if template in log_templates:
            log_id = log_templates[template]
        else:
            log_id = len(log_templates)
            log_templates[template] = log_id

        exists = False

        for node in nodes:
            if log_id in node.state.log_ids:
                parent = node
                nodes = node.get_children()
                exists = True
                continue

        if not exists:
            child = PrefixTree(State([log_id]), parent)
            parent.add_child(child)
            nodes = child.get_children()
            parent = child

    return prefix_tree


def main():
    prefix_tree, log_templates = \
        generate_prefix_tree("../resources/traces/",
                             "../resources/config.json")
    Visualizer(prefix_tree).visualize()
    pprint(log_templates)


if __name__ == '__main__':
    main()
