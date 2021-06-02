from whatthelog.definitions import PROJECT_ROOT
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.prefixtree.visualizer import Visualizer


def main():
    prefix_tree_pickle_path = PROJECT_ROOT.joinpath("resources/prefix_tree.pickle")
    pt = PrefixTreeFactory().unpickle_tree(prefix_tree_pickle_path)

    Visualizer(pt).visualize("prefix_tree_original.png")


if __name__ == '__main__':
    main()