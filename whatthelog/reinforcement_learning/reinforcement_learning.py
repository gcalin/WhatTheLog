import os
from pathlib import Path

from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.prefixtree.visualizer import Visualizer
from whatthelog.definitions import PROJECT_ROOT
# PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent.parent

if __name__ == '__main__':

    pt = PrefixTreeFactory().get_prefix_tree(PROJECT_ROOT.joinpath("resources/traces"), PROJECT_ROOT.joinpath("resources/config.json"))
    # Visualizer(pt).visualize("initial_pt.png")

    pt.remove_loops(False)
    # Visualizer(pt).visualize("after_pt.png")
