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

from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.auto_printer import AutoPrinter
from whatthelog.utils import bytes_tostring

def print(msg): AutoPrinter.static_print(msg)


#****************************************************************************************************
# Main Code
#****************************************************************************************************

def main():

    project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent
    tree_file = project_root.joinpath('out/fullPrefixTree.p')
    print(f"Pickled file weighs: {bytes_tostring(tree_file.stat().st_size)}")
    tree = PrefixTreeFactory.unpickle_tree(tree_file)
    print(f"Unpickled tree weighs: {bytes_tostring(asizeof(tree))}")

    states = tree.states
    states_size = sum([sys.getsizeof(state) for state in states])
    print(f"Tree has {len(states)} states "
          f"with total size {bytes_tostring(states_size)} "
          f"(average {bytes_tostring(states_size/len(states))})")

    properties = set([state.properties for state in states])
    props_size = sum([asizeof(prop) for prop in properties])
    print(f"Tree has {len(properties)} state properties "
          f"with total size {bytes_tostring(props_size)} "
          f"(average {bytes_tostring(props_size/len(properties))})")

    print(f"The edges matrix weighs: {bytes_tostring(asizeof(tree.edges))}")
    print(f"The list of states weighs: {bytes_tostring(asizeof(tree.states))}")
    print(f"The state hash index map weighs: {bytes_tostring(asizeof(tree.state_indices_by_hash))}")

if __name__ == "__main__":
    main()
