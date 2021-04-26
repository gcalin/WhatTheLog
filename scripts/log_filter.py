# -*- coding: utf-8 -*-
"""
Created on Thursday 04/22/2021
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
import math
from multiprocessing import Pool
from os import path
import sys
from time import time
import tracemalloc
from tqdm import tqdm
from typing import Union

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.syntaxtree.parser import Parser
from whatthelog.syntaxtree.syntax_tree import SyntaxTree


#****************************************************************************************************
# Global Variables
#****************************************************************************************************

pool_size_default = 8
chunk_size_default = 300000

#****************************************************************************************************
# Utility Functions
#****************************************************************************************************

def check_line(tree: SyntaxTree, line: str) -> Union[str, None]:
    return None if tree.search(line) else line

def get_peak_mem(snapshot, key_type='lineno') -> int:
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    return sum(stat.size for stat in snapshot.statistics(key_type))

# --- Parser for human-readable file sizes ---
def bytes_tostring(num): # Source: https://web.archive.org/web/20111010015624/http://blogmag.net/blog/read/38/Print_human_readable_file_size
    for x in ['bytes','KB','MB','GB','TB']:
        if num < 1024.0:
            return "%3.1f%s" % (num, x)
        num /= 1024.0

# --- Block splitter for counting line endings ---
def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b


#****************************************************************************************************
# Main Code
#****************************************************************************************************

def main(argv):

    start_time = time()
    tracemalloc.start()

    assert len(argv) >= 3, "Not enough arguments supplied!"

    # --- Parse CLI args ---
    config_filename = argv[0]
    log_filename = argv[1]
    output_filename = argv[2]
    subprocesses = int(argv[3] if len(argv) > 3 else pool_size_default)
    chunk_size = int(argv[4] if len(argv) > 4 else chunk_size_default)

    assert path.exists(config_filename), "Config file not found!"
    assert path.exists(log_filename), "Input file not found!"

    # --- Parse prefix tree ---
    print("[ Log Filter ] - Parsing configuration file...")
    parser = Parser()
    tree = parser.parse_file(config_filename)

    # --- Count lines in file ---
    print("[ Log Filter ] - Parsing logs file...")
    with open(log_filename, 'r') as f:

        line_total = sum(bl.count("\n") for bl in blocks(f))
        print(f"[ Log Filter ] - Counted {line_total} lines")

    # --- Run filtering ---
    print("[ Log Filter ] - Filtering logs...")
    with open(log_filename, 'r') as f:

        output = []
        finished = False
        pbar = tqdm(total=math.ceil(line_total/chunk_size), file=sys.stdout, leave=False)
        while not finished:

            # --- Parse chunk of lines ---
            slice = []
            for x in range(chunk_size):
                try:
                    slice.append(str(next(f)))
                except StopIteration:
                    finished = True
                    break

            # --- Filter chunk in subprocesses ---
            with Pool(subprocesses) as p:
                zipped = [(tree, line) for line in slice]
                output += p.starmap(check_line, zipped)

            if finished:
                pbar.close()
            else:
                pbar.update(1)

    # --- Remove nulls from result ---
    print("[ Log Filter ] - Removing filtered logs...")
    result = [x for x in output if x is not None]

    # --- Write output ---
    print("[ Log Filter ] - Writing output to file...")
    with open(output_filename, 'w+') as f:
        f.writelines(result)

    print(f"[ Log Filter ] - Done!")
    print(f"[ Log Filter ] - Time elapsed: {timedelta(seconds=time() - start_time)}")

    snapshot = tracemalloc.take_snapshot()
    total = get_peak_mem(snapshot)
    print(f"[ Log Filter ] - Peak memory usage: {bytes_tostring(total)}")


if __name__ == "__main__":
    main(sys.argv[1:])