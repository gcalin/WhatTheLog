#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from datetime import timedelta
import os
from time import time
import tracemalloc

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.prefixtree.visualizer import Visualizer
from whatthelog.auto_printer import AutoPrinter
from whatthelog.utils import get_peak_mem, bytes_tostring

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

    print(f"Done!")
    print(f"Time elapsed: {timedelta(seconds=time() - start_time)}")

    snapshot = tracemalloc.take_snapshot()
    total = get_peak_mem(snapshot)
    print(f"Peak memory usage: {bytes_tostring(total)}")

    Visualizer(pt).visualize()
