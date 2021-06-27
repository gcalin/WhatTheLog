#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
    specificity, recall = factory.eval_model(factory.evaluator)
    print(f"initial size: {len(pt)}, initial specificity: {specificity}, initial recall: {recall}")

    dendrogram = factory.get_dendrogram()
    print(f"Generated dendrogram with length: {len(dendrogram)}")

    accuracies = factory.eval_merges(dendrogram, 1000, 10)
    print(f"Evaluating {len(factory.tree)}-node model...")
    specificity, recall = factory.eval_model(factory.evaluator, debug=True)
    print(f"specificity={specificity}, recall={recall}")
    print(f"Done! Time elapsed: {timedelta(seconds=time() - start_time)}")
    print(f"final size: {len(factory.tree)}")