#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from datetime import timedelta
import os
from pathlib import Path
from time import time

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.auto_printer import AutoPrinter

def print(msg): AutoPrinter.static_print(msg)


#****************************************************************************************************
# Main Code
#****************************************************************************************************

if __name__ == '__main__':

    project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent

    start_time = time()

    pt = PrefixTreeFactory.get_prefix_tree(str(project_root.joinpath('resources/train_set')),
                                           str(project_root.joinpath('resources/config.json')), True)

    PrefixTreeFactory.pickle_tree(pt, project_root.joinpath('out/testPrefixTree.p'))

    print(f"Done! Parsed full tree of size: {len(pt)}")
    print(f"Time elapsed: {timedelta(seconds=time() - start_time)}")
