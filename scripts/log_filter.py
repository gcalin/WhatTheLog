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

from multiprocessing import Pool
from os import path
import sys
from time import time
from tqdm import tqdm
from typing import Union

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.parser import Parser
from whatthelog.prefixtree.prefix_tree import PrefixTree


#****************************************************************************************************
# Global Variables
#****************************************************************************************************

default_folds = 5


#****************************************************************************************************
# Main Code
#****************************************************************************************************

def check_line(tree: PrefixTree, line: str) -> Union[str, None]:
    return None if tree.search(line) else line

def main(argv):

    start_time = time()

    assert len(argv) >= 3, "Not enough arguments supplied!"

    config_filename = argv[0]
    log_filename = argv[1]
    output_filename = argv[2]
    subprocesses = argv[3] if len(argv) > 3 else default_folds

    assert path.exists(config_filename), "Config file not found!"
    assert path.exists(log_filename), "Input file not found!"

    print("[ Log Filter ] - Parsing configuration file...")

    parser = Parser()
    tree = parser.parse_file(config_filename)

    print("[ Log Filter ] - Reading logs...")

    with open(log_filename, 'r') as f:
        unfiltered_lines = f.readlines()

    print("[ Log Filter ] - Filtering logs...")

    # --- Run multiprocess ---
    with Pool(subprocesses) as p:
        zipped = [(tree, line) for line in unfiltered_lines]
        result = p.starmap(check_line, tqdm(zipped, file=sys.stdout, leave=False))

    print("[ Log Filter ] - Removing filtered logs...")

    result = filter(lambda x: x is not None, tqdm(result, file=sys.stdout, leave=False))

    print("[ Log Filter ] - Writing output to file...")

    with open(output_filename, 'w+') as f:
        f.writelines(result)

    end_time = time()
    print(f"[ Log Filter ] - Done! Time elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main(sys.argv[1:])