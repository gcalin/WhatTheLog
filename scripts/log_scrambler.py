# -*- coding: utf-8 -*-
"""
Created on Thursday 04/26/2021
Author: Tommaso Brandirali
Email: tommaso.brandirali@gmail.com
"""

#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

sys.path.insert(0, "./../")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.syntaxtree.parser import Parser
from whatthelog.syntaxtree.syntax_tree import SyntaxTree
from whatthelog.auto_printer import AutoPrinter
from whatthelog.utils import get_peak_mem, bytes_tostring

#****************************************************************************************************
# Global Variables
#****************************************************************************************************

pool_size_default = 8
config_default = os.path.join(pathlib.Path(__file__).parent.absolute(), "../resources/config.json")

def print(msg): AutoPrinter.static_print(msg)

#****************************************************************************************************
# Utility Functions
#****************************************************************************************************

# --- Parses a section of lines adjacent to the input line which have the same log format ---
def get_section(lines: List[str], line: int, tree: SyntaxTree) -> List[int]:

    node = tree.search(lines[line])
    section = [line]

    counter = line + 1
    while counter < len(lines) - 1 and tree.search(lines[counter]) == node:
        section.append(counter)
        counter += 1

    counter = line - 1
    while counter >= 0 and tree.search(lines[counter]) == node:
        section.insert(0, counter)
        counter -= 1

    return section


#****************************************************************************************************
# Main Code
#****************************************************************************************************

def delete_one(lines: List[str], tree: SyntaxTree):

    line = random.choice(range(len(lines)))
    section = get_section(lines, line, tree)
    section.reverse()

    for l in section:
        del lines[l]

def swap(lines: List[str], tree: SyntaxTree):

    elem = random.choice(range(len(lines)-1))
    section1 = get_section(lines, elem, tree)

    next_elem = section1[-1] + 1 if section1[-1] + 1 < len(lines) - 1 else section1[0] - 1
    section2 = get_section(lines, next_elem, tree)

    lines[section1[0]:section1[-1]+1], lines[section2[0]:section2[-1]+1] = \
        lines[section2[0]:section2[-1]+1], lines[section1[0]:section1[-1]+1]

def r_swap(lines: List[str], tree: SyntaxTree):

    elem1 = random.choice(range(len(lines)))
    section1 = get_section(lines, elem1, tree)

    elem2 = random.choice(range(len(lines)))
    while elem2 in section1:
        elem2 = random.choice(range(len(lines)))
    section2 = get_section(lines, elem2, tree)

    lines[section1[0]:section1[-1]+1], lines[section2[0]:section2[-1]+1] = \
        lines[section2[0]:section2[-1]+1], lines[section1[0]:section1[-1]+1]

def process_file(input_file: str, output_file: str, tree: SyntaxTree) -> None:

    with open(input_file, 'r') as f:

        lines = f.readlines()
        n_mutations = random.randint(1, 3)
        mutations = [random.choice([delete_one, swap, r_swap]) for _ in range(n_mutations)]

        for func in mutations:
            func(lines, tree)

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
    parser = Parser()
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
