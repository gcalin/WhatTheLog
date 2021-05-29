# -*- coding: utf-8 -*-
"""
Created on Thursday 04/26/2021
Author: Tommaso Brandirali
Email: tommaso.brandirali@gmail.com
"""

# ****************************************************************************************************
# Imports
# ****************************************************************************************************

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from datetime import timedelta
from multiprocessing import Pool
import os
import pathlib
import random
import sys
from time import time
from typing import List
import tracemalloc
from tqdm import tqdm

from scripts.match_trace import match_trace
from whatthelog.prefixtree.prefix_tree import PrefixTree

sys.path.insert(0, "./../")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.syntaxtree.syntax_tree_factory import SyntaxTreeFactory
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.auto_printer import AutoPrinter
from whatthelog.utils import get_peak_mem, bytes_tostring

# ****************************************************************************************************
# Global Variables
# ****************************************************************************************************

pool_size_default = 8
config_default = os.path.join(pathlib.Path(__file__).parent.absolute(), "../resources/config.json")
# random.seed(os.environ['random_seed'] if 'random_seed' in os.environ else 5)


def print(msg): AutoPrinter.static_print(msg)


# ****************************************************************************************************
# Utility Functions
# ****************************************************************************************************

# --- Parses a section of lines adjacent to the input line which have the same log format ---
def get_section(lines: List[str], line: int, tree: SyntaxTree) -> List[int]:
    # If the line is not in the a valid index, no section to find
    if line < 0 or line >= len(lines):
        return []
    # TODO: Not in tree?
    # Find the corresponding leaf in the prefix tree
    node = tree.search(lines[line])

    # Initialize the section
    # TODO: Tuple representation?
    section = [line]

    counter = line + 1
    # Collect all the adjacent lines to the below it that are of the same state
    while counter < len(lines) and tree.search(lines[counter]) == node:
        section.append(counter)
        counter += 1

    counter = line - 1
    # Collect all the adjacent lines to the above it that are of the same state
    while counter >= 0 and tree.search(lines[counter]) == node:
        section.insert(0, counter)
        counter -= 1

    return section


# ****************************************************************************************************
# Main Code
# ****************************************************************************************************

def delete_one(lines: List[str], tree: SyntaxTree):
    """
    Deletes the section corresponding to a random line in a list.
    :param lines: the list of lines to delete from.
    :param tree: the syntax tree that defines which lines form a section.
    """

    # If no lines exist, nothing to delete
    if len(lines) == 0:
        return

    # Randomly choose an index in the list
    # TODO: More efficiently?
    line = random.choice(range(len(lines)))

    # Get the corresponding section
    section = get_section(lines, line, tree)

    # Reverse the order of the indexes such that
    # Elements are accessed from highest to lowest
    # To avoid out of bounds exceptions
    section.reverse()

    # Delete each line from the list
    for l in section:
        del lines[l]


def swap(lines: List[str], tree: SyntaxTree):
    """
    Swaps two adjacent sections corresponding to a random line in a list and its neighbour.
    :param lines: the list of lines to delete from.
    :param tree: the syntax tree that defines which lines form a section.
    """

    # If fewer than 1 section, nothing to swap
    if len(lines) <= 1 or len(get_section(lines, 0, tree)) == len(lines):
        return

    # Get a random position from the list
    elem = random.choice(range(len(lines) - 1))

    # Get the corresponding section
    section1 = get_section(lines, elem, tree)

    # Get either the next section, if there exists one or the previous one
    next_elem = section1[-1] + 1 if section1[-1] + 1 < len(lines) - 1 else section1[0] - 1

    # Find the corresponding section
    section2 = get_section(lines, next_elem, tree)

    # If the previous section was chosen, swap the two such that they are ordered
    if section1[0] > section2[0]:
        section1, section2 = section2, section1

    # Swap the positions of the elements of the two sections, keep everything else in the same order
    lines[:] = lines[:section1[0]] \
               + lines[section2[0]:section2[-1] + 1] \
               + lines[section1[-1] + 1:section2[0]] \
               + lines[section1[0]:section1[-1] + 1] \
               + lines[section2[-1] + 1:]


def r_swap(lines: List[str], tree: SyntaxTree):
    """
    Swaps two disjoint sections corresponding to two random lines in a list.
    TODO: Inject mocks for testing?
    :param lines: the list of lines to delete from.
    :param tree: the syntax tree that defines which lines form a section.
    """
    # If fewer than 1 section, nothing to swap
    if len(lines) <= 1 or len(get_section(lines, 0, tree)) == len(lines):
        return

    # Get a random position from the list and its corresponding section
    elem1 = random.choice(range(len(lines)))
    section1 = get_section(lines, elem1, tree)

    # Get another random position from the list, that is not part of the previous find section
    elem2 = random.choice(range(len(lines)))
    while elem2 in section1:
        elem2 = random.choice(range(len(lines)))

    # Get the second corresponding section
    section2 = get_section(lines, elem2, tree)

    lines[:] = lines[:section1[0]] \
               + lines[section2[0]:section2[-1] + 1] \
               + lines[section1[-1] + 1:section2[0]] \
               + lines[section1[0]:section1[-1] + 1] \
               + lines[section2[-1] + 1:]


def process_file(input_file: str, output_file: str, tree: SyntaxTree) -> None:
    with open(input_file, 'r') as f:
        lines = f.readlines()
        n_mutations = random.randint(1, 3)
        mutations = [random.choice([delete_one, swap, r_swap]) for _ in range(n_mutations)]

        for func in mutations:
            func(lines, tree)

    with open(output_file, 'w+') as f:
        f.writelines(lines)


def produce_false_trace(input_file: str, output_file: str, syntax_tree: SyntaxTree, state_model: PrefixTree) -> None:
    """
    Produces a log that guarantees a false trace will be created.
    :param input_file: The file containing the trace.
    :param output_file: The file in which the false trace should be written.
    :param syntax_tree: The syntax tree used to match an individual log entry.
    :param state_model: The state model used to validate a trace.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
        n_mutations = random.randint(1, 3)
        mutations = [random.choice([delete_one, swap, r_swap]) for _ in range(n_mutations)]

        # Apply random mutations
        for func in mutations:
            func(lines, syntax_tree)

        # While the produced log is still accepted, continue scrambling it
        while match_trace(state_model, lines.copy(), syntax_tree):
            n_mutations = random.randint(1, 3)
            mutations = [random.choice([delete_one, swap, r_swap]) for _ in range(n_mutations)]

            # Apply random mutations
            for func in mutations:
                func(lines, syntax_tree)

    # Write to the output file
    with open(output_file, 'w+') as f:
        f.writelines(lines)


def main(argv):
    start_time = time()
    tracemalloc.start()

    assert len(argv) > 1, "Not enough arguments supplied!"

    input_dir = argv[0]
    output_dir = argv[1]
    config_file = argv[2] if len(argv) > 2 else config_default
    pool_size = argv[3] if len(argv) > 3 else pool_size_default

    assert os.path.isdir(input_dir), "Cannot find input directory!"
    assert os.path.isdir(output_dir), "Cannot find output directory!"
    assert os.path.isfile(config_file), "Cannot find config file!"

    files = [os.path.join(input_dir, name)
             for name in os.listdir(input_dir)
             if os.path.isfile(os.path.join(input_dir, name))]

    # --- Parse prefix tree ---
    print("[ Log Filter ] - Parsing configuration file...")
    parser = SyntaxTreeFactory()
    tree = parser.parse_file(config_file)

    # --- Run filtering ---
    print("[ Log Filter ] - Filtering logs...")
    finished = False
    pbar = tqdm(total=len(files), file=sys.stdout, leave=False)
    while not finished:

        curr_n_files = min(pool_size, len(files))
        curr_files = [files.pop(0) for _ in range(curr_n_files)]

        if curr_n_files < pool_size:
            finished = True

        output_files = [os.path.join(output_dir, os.path.basename(name)) for name in curr_files]

        # --- Filter chunk in subprocesses ---
        with Pool(curr_n_files) as p:
            p.starmap(process_file, zip(curr_files, output_files, [tree for _ in range(curr_n_files)]))

        pbar.update(curr_n_files)

    print(f"[ Log Filter ] - Done!")
    print(f"[ Log Filter ] - Time elapsed: {timedelta(seconds=time() - start_time)}")

    snapshot = tracemalloc.take_snapshot()
    total = get_peak_mem(snapshot)
    print(f"[ Log Filter ] - Peak memory usage: {bytes_tostring(total)}")


if __name__ == "__main__":
    main(sys.argv[1:])
