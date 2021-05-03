#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from datetime import timedelta
import os
from pathlib import Path
import sys
from time import time
import tracemalloc

sys.path.insert(0, "./../")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.auto_printer import AutoPrinter
from whatthelog.utils import profile_mem

def print(msg): AutoPrinter.static_print(msg)


#****************************************************************************************************
# Main Code
#****************************************************************************************************

if __name__ == '__main__':

    project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent

    start_time = time()
    # tracemalloc.start()

    pt = PrefixTreeFactory.get_prefix_tree(str(project_root.joinpath('out/traces')),
                                           str(project_root.joinpath('resources/config.json')))

    PrefixTreeFactory.pickle_tree(pt, project_root.joinpath('out/fullPrefixTree.p'))

    print(f"Done! Parsed full tree of size: {pt.size()}")
    print(f"Time elapsed: {timedelta(seconds=time() - start_time)}")

    # snapshot = tracemalloc.take_snapshot()
    # profile_mem(snapshot)
