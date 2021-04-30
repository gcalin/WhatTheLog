#****************************************************************************************************
# Imports
#****************************************************************************************************

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# External
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
from pathlib import Path
from pympler.asizeof import asizeof
import sys

sys.path.insert(0, "./../")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Internal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from whatthelog.prefixtree.state import State
from whatthelog.prefixtree.edge import Edge
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.auto_printer import AutoPrinter
from whatthelog.utils import bytes_tostring

def print(msg): AutoPrinter.static_print(msg)


#****************************************************************************************************
# Main Code
#****************************************************************************************************

def main():

    print(f"State weighs: {bytes_tostring(asizeof(State([])))}")
    print(f"Edge weighs: {bytes_tostring(asizeof(Edge(None, None)))}")

    # project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent
    # tree_file = project_root.joinpath('out/fullPrefixTree.p')
    # print(f"Pickled file weighs: {bytes_tostring(tree_file.stat().st_size)}")
    # tree = PrefixTreeFactory.unpickle_tree(tree_file)
    # print(f"Unpickled tree weighs: {bytes_tostring(asizeof(tree))}")
    # edge_list = list(tree.edges)
    # edges_size = sum([asizeof(edge) for edge in edge_list])
    # print(f"Tree has {len(edge_list)} edges "
    #       f"with total size {bytes_tostring(edges_size)} "
    #       f"(average {bytes_tostring(edges_size/len(edge_list))})")
    # states = tree.states
    # print(f"Tree states dictionary weighs: {bytes_tostring(asizeof(states))}")

if __name__ == "__main__":
    main()
