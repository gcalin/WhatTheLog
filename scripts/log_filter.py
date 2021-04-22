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

from os import path
import sys
from tqdm import tqdm

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.parser import Parser


#****************************************************************************************************
# Main Code
#****************************************************************************************************

def main(argv):

    assert len(argv) > 3, "Not enough arguments supplied!"

    config_filename = argv[0]
    log_filename = argv[1]
    output_filename = argv[2]

    assert path.exists(config_filename), "Config file not found!"
    assert path.exists(log_filename), "Input file not found!"
    assert path.exists(output_filename), "Output file not found!"

    print("[ Log Filter ] - Parsing configuration file...")

    parser = Parser()
    tree = parser.parse_file(config_filename)

    print("[ Log Filter ] - Reading logs...")

    with open(log_filename, 'r') as f:
        unfiltered_lines = f.readlines()

    print("[ Log Filter ] - Filtering logs...")

    filtered_lines = []
    pbar = tqdm(unfiltered_lines, file=sys.stdout, leave=False)
    for line in pbar:
        if not tree.search(line):
            filtered_lines.append(line)

    print("[ Log Filter ] - Writing output to file...")

    with open(output_filename, 'w') as f:
        f.writelines(filtered_lines)

    print("[ Log Filter ] - Done!")


if __name__ == "__main__":
    main(sys.argv[1:])