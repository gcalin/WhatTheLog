import os

from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.prefixtree.visualizer import Visualizer


if __name__ == '__main__':
    path = os.path.abspath(os.path.dirname(__file__))

    pt = PrefixTreeFactory.get_prefix_tree(path + '/../resources/traces/',
                                           path + '/../resources/config.json')
    Visualizer(pt).visualize()
