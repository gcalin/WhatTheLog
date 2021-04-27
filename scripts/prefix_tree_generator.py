#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from math import floor
from multiprocessing import Pool
import os
from pprint import pprint
import sys
from time import time
import tracemalloc
from typing import List, Dict, Tuple
from tqdm import tqdm

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.exceptions.unidentified_log_exception import \
    UnidentifiedLogException
from whatthelog.prefixtree.prefix_tree import PrefixTree
from whatthelog.prefixtree.state import State
from whatthelog.prefixtree.visualizer import Visualizer
from whatthelog.syntaxtree.parser import Parser
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.auto_printer import AutoPrinter
from whatthelog.utils import get_peak_mem, bytes_tostring, blocks

#****************************************************************************************************
# Global Variables
#****************************************************************************************************

pool_size_default = 8
def print(msg): AutoPrinter.static_print(msg)


#****************************************************************************************************
# Prefix Tree Factory
#****************************************************************************************************

# def generate_prefix_tree(log_dir: str, config_file: str) -> \
#         Tuple[PrefixTree, Dict[str, int]]:
#     """
#     Script to parse log file into prefix tree.
#
#     :param log_dir: Path to directory containing traces.
#     :param config_file: Path to configuration file for syntax tree
#     :return: Prefix tree along with a dictionary mapping log templates
#      to unique ids.
#     """
#     syntax_tree = Parser().parse_file(config_file)
#     prefix_tree = PrefixTree(State([0]), None)
#     log_templates = {"": 0}
#
#     pbar = tqdm(total=len(os.listdir(log_dir)), file=sys.stdout, leave=False)
#     for filename in os.listdir(log_dir):
#         with open(log_dir + filename, 'r') as f:
#             logs = [log.strip() for log in f.readlines()]
#
#         parse_trace(logs, log_templates, syntax_tree, prefix_tree)
#         pbar.update(1)
#     return prefix_tree, log_templates
#
#
# def parse_trace(logs: List[str],
#                 log_templates: Dict[str, int],
#                 syntax_tree: SyntaxTree,
#                 prefix_tree: PrefixTree) -> PrefixTree:
#     """
#     Function that parses a trace file and modifies the given
#     prefix tree to include it.
#
#     :param logs: A list of logs representing this trace
#     :param log_templates: The mapping between log templates and log ids
#     :param syntax_tree: The syntax tree used to get the log template from
#     the log
#     :param prefix_tree: The current prefix tree to be used
#     :return: None
#     """
#
#     parent = prefix_tree
#     nodes = prefix_tree.get_children()
#
#     for log in logs:
#         template = syntax_tree.search(log).name
#         if template is None:
#             raise UnidentifiedLogException(
#                 log + " was not identified as a valid log.")
#
#         if template in log_templates:
#             log_id = log_templates[template]
#         else:
#             log_id = len(log_templates)
#             log_templates[template] = log_id
#
#         exists = False
#
#         for node in nodes:
#             if log_id in node.state.log_ids:
#                 parent = node
#                 nodes = node.get_children()
#                 exists = True
#                 continue
#
#         if not exists:
#             child = PrefixTree(State([log_id]), parent)
#             parent.add_child(child)
#             nodes = child.get_children()
#             parent = child
#
#     return prefix_tree

def parse_trace_async(trace_file: str, syntax_tree: SyntaxTree) -> PrefixTree:

    root = PrefixTree(State([]), None)
    with open(trace_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    curr_node = root
    for line in lines:

        syntax_node = syntax_tree.search(line)
        if syntax_node is None:
            raise UnidentifiedLogException(line + " was not identified as a valid log.")

        new_node = PrefixTree(State([syntax_node.get_pattern()]), curr_node)
        curr_node.add_child(new_node)
        curr_node = new_node

    print(f"Tree depth = {root.depth()}")

    return root


def parse_tree(log_dir: str, config_file: str, pool_size: int = pool_size_default) -> PrefixTree:

    assert os.path.isdir(log_dir), "Log directory not found!"
    assert os.path.isfile(config_file), "Config file not found!"

    print("Parsing syntax tree...")

    syntax_tree = Parser().parse_file(config_file)
    trace_files = [os.path.join(log_dir, tracefile) for tracefile in os.listdir(log_dir)]

    # --- Count lines in file ---
    print("Parsing trace lengths...")
    max_length = 0
    for log_filename in trace_files:
        with open(log_filename, 'r') as f:

            line_total = sum(bl.count("\n") for bl in blocks(f))
            if line_total > max_length:
                max_length = line_total

    # --- Dynamically set recursion limit ---
    print(f"Maximum trace length found: {max_length}, maximum recursion limit: {sys.getrecursionlimit()}")
    if sys.getrecursionlimit() <= int(max_length * 1.1):
        print(f"Setting recursion limit to {int(max_length * 1.1)}")
        sys.setrecursionlimit(int(max_length * 1.1))

    # --- Parse individual traces into separate trees ---
    print("Parsing individual log traces...")
    inputs = list(zip(trace_files, [syntax_tree for _ in range(len(trace_files))]))
    pbar = tqdm(inputs, total=len(inputs), file=sys.stdout, leave=False)
    with Pool(pool_size) as p:
        trees = p.starmap(parse_trace_async, pbar, chunksize=2)

    print("Merging trees...")

    # --- Merge trees in parallel ---
    total = len(trees)
    pbar = tqdm(total=total, file=sys.stdout, leave=False)
    while len(trees) > 1:

        threads = max(floor(len(trees)/2), pool_size)
        source_trees = [(trees.pop(0), trees.pop(0)) for _ in range(threads)]

        with ThreadPoolExecutor(max_workers=threads) as executor:
            trees += executor.map(lambda tup: tup[0].merge(tup[1]), source_trees)

        pbar.update(total - len(trees))

    assert len(trees) == 1, f"Something went wrong: {len(trees)} found after merging!"

    return trees[0]


def main():

    start_time = time()
    tracemalloc.start()

    # prefix_tree, log_templates = generate_prefix_tree("../resources/traces/", "../resources/config.json")
    prefix_tree = parse_tree("../out/traces_test/", "../resources/config.json", 10)

    print(f"Done!")
    print(f"Time elapsed: {timedelta(seconds=time() - start_time)}")

    snapshot = tracemalloc.take_snapshot()
    total = get_peak_mem(snapshot)
    print(f"Peak memory usage: {bytes_tostring(total)}")

    # print("Building visualization...")
    # Visualizer(prefix_tree).visualize()
    # pprint(log_templates)


if __name__ == '__main__':
    main()
