import os
import sys
from typing import List
from tqdm import tqdm

from whatthelog.prefixtree.prefix_tree_graph import PrefixTreeGraph
from whatthelog.prefixtree.state import State
from whatthelog.syntaxtree.parser import Parser
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.auto_printer import AutoPrinter

pool_size_default = 8

def print(msg): AutoPrinter.static_print(msg)


class PrefixTreeFactory(AutoPrinter):
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

        pbar = tqdm(os.listdir(log_dir), file=sys.stdout)
        for filename in pbar:
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
        :param syntax_tree: The syntax tree used to get the log template from the log
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

    # @staticmethod
    # def parse_trace_async(trace_file: str, syntax_tree: SyntaxTree) -> PrefixTreeGraph:
    #
    #     root = State([""])
    #     tree = PrefixTreeGraph(root)
    #
    #     with open(trace_file, 'r') as f:
    #         lines = [line.strip() for line in f.readlines()]
    #
    #     curr_state = root
    #     for line in lines:
    #
    #         syntax_node = syntax_tree.search(line)
    #         if syntax_node is None:
    #             raise UnidentifiedLogException(line + " was not identified as a valid log.")
    #
    #         new_state = State([syntax_node.get_pattern()])
    #         tree.add_child(new_state, curr_state)
    #         curr_state = new_state
    #
    #     return tree
    #
    # @staticmethod
    # def parse_tree(log_dir: str, config_file: str, pool_size: int = pool_size_default) -> PrefixTreeGraph:
    #
    #     assert os.path.isdir(log_dir), "Log directory not found!"
    #     assert os.path.isfile(config_file), "Config file not found!"
    #
    #     print("Parsing syntax tree...")
    #
    #     syntax_tree = Parser().parse_file(config_file)
    #     trace_files = [os.path.join(log_dir, tracefile) for tracefile in os.listdir(log_dir)]
    #
    #     # --- Parse individual traces into separate trees ---
    #     print("Parsing individual log traces...")
    #     inputs = list(zip(trace_files, [syntax_tree for _ in range(len(trace_files))]))
    #     pbar = tqdm(inputs, total=len(inputs), file=sys.stdout, leave=False)
    #     with Pool(pool_size) as p:
    #         trees = p.starmap(PrefixTreeFactory.parse_trace_async, pbar, chunksize=2)
    #
    #     print("Merging trees...")
    #
    #     # --- Merge trees in parallel ---
    #     total = len(trees)
    #     pbar = tqdm(total=total, file=sys.stdout, leave=False)
    #     while len(trees) > 1:
    #
    #         threads = max(floor(len(trees)/2), pool_size)
    #         source_trees = [(trees.pop(0), trees.pop(0)) for _ in range(threads)]
    #
    #         with ThreadPoolExecutor(max_workers=threads) as executor:
    #             trees += executor.map(lambda tup: tup[0].merge(tup[1]), source_trees)
    #
    #         pbar.update(total - len(trees))
    #
    #     assert len(trees) == 1, f"Something went wrong: {len(trees)} found after merging!"
    #
    #     return trees[0]


class UnidentifiedLogException(Exception):
    pass
