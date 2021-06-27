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

from whatthelog.clustering.state_model_factory import StateModelFactory
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.auto_printer import AutoPrinter
from whatthelog.utils import group

def print(msg): AutoPrinter.static_print(msg)


#****************************************************************************************************
# Main Code
#****************************************************************************************************

if __name__ == '__main__':

    setrecursionlimit(int((10**6)/2))

    project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent

    traces_dir = project_root.joinpath('resources/traces')
    config_file = project_root.joinpath('resources/config.json')
    results_dir = project_root.joinpath('out/results')
    iterations = 20
    traces_per_iter = 140

    start_time = time()
    print("Fetching trace sets...")

    # Fetch and group traces
    traces = os.listdir(traces_dir)[100:]
    traces = traces[:(iterations * traces_per_iter)]
    traces = [traces_dir.joinpath(file) for file in traces]
    trace_sets = group(traces, traces_per_iter)
    print(f"Fetched {len(trace_sets)} sets of {len(trace_sets[0])} traces")

    reductions = []
    for i, trace_set in enumerate(trace_sets):

        print(f"Running iteration {i+1}...")
        volume = sum([len(open(trace, 'r').readlines()) for trace in trace_set])
        print(f"Iteration volume: {volume} logs")

        pt = PrefixTreeFactory.get_prefix_tree(None, config_file, True, trace_set)
        factory = StateModelFactory(pt, skip_cache_build=True)
        initial_size = len(pt)
        dendrogram = factory.get_dendrogram()
        factory.merge_full(dendrogram)
        reductions.append(initial_size / len(factory.tree))

    print(f"Average reduction = {sum(reductions)/len(reductions)}")
    print(f"Done! Time elapsed: {timedelta(seconds=time() - start_time)}")