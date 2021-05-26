#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from copy import deepcopy
from datetime import timedelta
import os
from pathlib import Path
from sys import setrecursionlimit
from time import time

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.clustering.state_model_factory import StateModelFactory
from whatthelog.auto_printer import AutoPrinter

def print(msg): AutoPrinter.static_print(msg)


#****************************************************************************************************
# Main Code
#****************************************************************************************************

if __name__ == '__main__':

    setrecursionlimit(int((10**6)/2))

    project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent
    start_time = time()

    pt = PrefixTreeFactory.unpickle_tree(project_root.joinpath('out/testPrefixTree.p'))

    factory = StateModelFactory(pt)
    model = factory.run_clustering('louvain')

    print(f"Done! Final model size: {len(model)}, time elapsed: {timedelta(seconds=time() - start_time)}")

    factory.pickle_model(model, project_root.joinpath('out/testModel.p'))
