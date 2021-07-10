from whatthelog.definitions import PROJECT_ROOT
from whatthelog.prefixtree.prefix_tree_factory import PrefixTreeFactory
from whatthelog.prefixtree.visualizer import Visualizer


def main():
    # prefix_tree_pickle_path = PROJECT_ROOT.joinpath("resources/prefix_tree.pickle")
    pt = PrefixTreeFactory().get_prefix_tree(PROJECT_ROOT.joinpath("resources/paper_examples"),
                                             PROJECT_ROOT.joinpath("resources/config.json"),
                                             unique_graph=True)

    Visualizer(pt).visualize("prefix_tree_unique.png")


if __name__ == '__main__':
    main()