#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from datetime import timedelta
import os
import pickle
import sys
from time import time
import tracemalloc

sys.path.insert(0, "./../")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.prefixtree.visualizer import Visualizer
from whatthelog.auto_printer import AutoPrinter
from whatthelog.utils import get_peak_mem, profile_mem, bytes_tostring

def print(msg): AutoPrinter.static_print(msg)


#****************************************************************************************************
# Main Code
#****************************************************************************************************

if __name__ == '__main__':

    path = os.path.abspath(os.path.dirname(__file__))

    start_time = time()
    tracemalloc.start()

    pt = PrefixTreeFactory.get_prefix_tree(path + '/../resources/traces/',
                                           path + '/../resources/config.json')

    with open(path + '/../out/fullPrefixTree.p', 'wb+') as file:
        pickle.dump(pt, file)

    print(f"Done! Parsed full tree of size: {pt.size()}")
    print(f"Time elapsed: {timedelta(seconds=time() - start_time)}")

    snapshot = tracemalloc.take_snapshot()
    profile_mem(snapshot)
    total = get_peak_mem(snapshot)
    print(f"Peak memory usage: {bytes_tostring(total)}")

    # Visualizer(pt).visualize()
