#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
from concurrent.futures.thread import ThreadPoolExecutor
from math import floor
from multiprocessing import Pool
import pickle
from typing import List
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

pool_size_default = 8
chunksize_default = 2

def print(msg): AutoPrinter.static_print(msg)


#****************************************************************************************************
# Prefix Tree Factory
#****************************************************************************************************

class PrefixTreeFactory(AutoPrinter):
    """
    Prefix tree factory class.
    """

    @staticmethod
    def get_prefix_tree(traces_dir: str, config_file_path: str,
                        pool_size: int = pool_size_default, chunk_size: int = chunksize_default) -> PrefixTree:
        """
        Parses a full tree from a set of log traces in a common directory,
        using a user-supplied syntax tree from an input configuration file.
        Assumes that all log traces are linear.
        :param traces_dir: the directory containing the log files to be parsed
        :param config_file_path: the configuration file describing the syntax tree
        :param pool_size: the maximum number of subprocesses and threads to use for processing,
                          default is 8
        :param chunk_size: the chunk size setting for the multiprocessed map,
                           default is 2
        :return: the full prefix tree
        """

        return PrefixTreeFactory.__parse_tree(traces_dir, config_file_path, pool_size, chunk_size)

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
        syntax_tree = Parser().parse_file(config_file)
        prefix_tree = PrefixTree(State([""]))

        pbar = tqdm(os.listdir(log_dir), file=sys.stdout)
        for filename in pbar:
            with open(log_dir + filename, 'r') as f:
                logs = [log.strip() for log in f.readlines()]

            PrefixTreeFactory.__parse_trace(logs, syntax_tree, prefix_tree)
        return prefix_tree

    @staticmethod
    def __parse_trace(logs: List[str],
                      syntax_tree: SyntaxTree,
                      prefix_tree: PrefixTree) -> PrefixTree:
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

    @staticmethod
    def __parse_trace_async(trace_file: str, syntax_tree: SyntaxTree) -> PrefixTree:

        root = State([""])
        tree = PrefixTree(root)

        with open(trace_file, 'r') as f:

            curr_state = root
            for line in f:

                line = line.strip()
                syntax_node = syntax_tree.search(line)
                if syntax_node is None:
                    raise UnidentifiedLogException(line + " was not identified as a valid log.")

                new_state = State([syntax_node.get_pattern()])
                tree.add_child(new_state, curr_state)
                curr_state = new_state

        return tree

    @staticmethod
    def __parse_tree(log_dir: str, config_file: str, pool_size: int, chunk_size: int) -> PrefixTree:

        if not os.path.isdir(log_dir):
            raise NotADirectoryError("Log directory not found!")
        if not os.path.isfile(config_file):
            raise FileNotFoundError("Config file not found!")

        print("Parsing syntax tree...")

        syntax_tree = Parser().parse_file(config_file)
        trace_files = [os.path.join(log_dir, tracefile)
                       for tracefile in os.listdir(log_dir)
                       if os.path.isfile(os.path.join(log_dir, tracefile))]

        # --- Parse individual traces into separate trees ---
        print("Parsing individual log traces...")
        inputs = list(zip(trace_files, [syntax_tree for _ in range(len(trace_files))]))
        pbar = tqdm(inputs, total=len(inputs), file=sys.stdout)
        with Pool(pool_size) as p:
            trees = p.starmap(PrefixTreeFactory.__parse_trace_async, pbar, chunksize=chunk_size)

        print("Merging trees...")

        # --- Merge trees in parallel ---
        total = len(trees)
        pbar = tqdm(total=total, file=sys.stdout, leave=False)
        while len(trees) > 1:

            threads = min(floor(len(trees)/2), pool_size)
            source_trees = [(trees.pop(0), trees.pop(0)) for _ in range(threads)]

            with ThreadPoolExecutor(max_workers=threads) as executor:
                executor.map(lambda tup: tup[0].merge(tup[1]), source_trees)

            trees += [dest for dest, source in source_trees]
            pbar.update(total - len(trees))

        assert len(trees) == 1, f"Something went wrong: {len(trees)} found after merging!"

        return trees[0]
